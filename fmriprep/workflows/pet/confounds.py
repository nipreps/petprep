# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright The NiPreps Developers <nipreps@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#
"""
Calculate PET confounds
^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_pet_confs_wf

"""

from nipype.algorithms import confounds as nac
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from templateflow.api import get as get_template

from ...config import DEFAULT_MEMORY_MIN_GB
from ...interfaces import DerivativesDataSink
from ...interfaces.confounds import (
    FilterDropped,
    PETSummary,
    FramewiseDisplacement,
    FSLMotionParams,
    FSLRMSDeviation,
    GatherConfounds,
)
from .outputs import prepare_timing_parameters

def init_pet_confs_wf(
    mem_gb: float,
    metadata: dict,
    regressors_all_comps: bool,
    regressors_dvars_th: float,
    regressors_fd_th: float,
    freesurfer: bool = False,
    name: str = 'pet_confs_wf',
):
    """
    Build a workflow to generate and write out confounding signals.

    This workflow calculates confounds for a PET series, and aggregates them
    into a :abbr:`TSV (tab-separated value)` file, for use as nuisance
    regressors in a :abbr:`GLM (general linear model)`.
    The following confounds are calculated, with column headings in parentheses:

    #. Region-wise average signal (``csf``, ``white_matter``, ``global_signal``)
    #. DVARS - original and standardized variants (``dvars``, ``std_dvars``)
    #. Framewise displacement, based on head-motion parameters
       (``framewise_displacement``)
    #. Cosine basis set for high-pass filtering w/ 0.008 Hz cut-off
       (``cosine_XX``)
    #. Non-steady-state volumes (``non_steady_state_XX``)
    #. Estimated head-motion parameters, in mm and rad
       (``trans_x``, ``trans_y``, ``trans_z``, ``rot_x``, ``rot_y``, ``rot_z``)


    Prior to estimating aCompCor and tCompCor, non-steady-state volumes are
    censored and high-pass filtered using a :abbr:`DCT (discrete cosine
    transform)` basis.
    The cosine basis, as well as one regressor per censored volume, are included
    for convenience.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from fmriprep.workflows.pet.confounds import init_pet_confs_wf
            wf = init_pet_confs_wf(
                mem_gb=1,
                metadata={},
                regressors_all_comps=False,
                regressors_dvars_th=1.5,
                regressors_fd_th=0.5,
            )

    Parameters
    ----------
    mem_gb : :obj:`float`
        Size of PET file in GB - please note that this size
        should be calculated after resamplings that may extend
        the FoV
    metadata : :obj:`dict`
        BIDS metadata for PET file
    name : :obj:`str`
        Name of workflow (default: ``pet_confs_wf``)
    regressors_all_comps : :obj:`bool`
        Indicates whether CompCor decompositions should return all
        components instead of the minimal number of components necessary
        to explain 50 percent of the variance in the decomposition mask.
    regressors_dvars_th : :obj:`float`
        Criterion for flagging DVARS outliers
    regressors_fd_th : :obj:`float`
        Criterion for flagging framewise displacement outliers

    Inputs
    ------
    pet
        PET image, after the prescribed corrections (STC, HMC and SDC)
        when available.
    pet_mask
        PET series mask
    motion_xfm
        ITK-formatted head motion transforms
    t1w_mask
        Mask of the skull-stripped template image
    t1w_tpms
        List of tissue probability maps in T1w space
    petref2anat_xfm
        Affine matrix that maps the PET reference space into alignment with
        the anatomical (T1w) space

    Outputs
    -------
    confounds_file
        TSV of all aggregated confounds
    rois_report
        Reportlet visualizing white-matter/CSF mask used for aCompCor,
        the ROI for tCompCor and the PET brain mask.
    confounds_metadata
        Confounds metadata dictionary.
    crown_mask
        Mask of brain edge voxels

    """
    from nireports.interfaces.nuisance import ConfoundsCorrelationPlot
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow
    from niworkflows.interfaces.confounds import ExpandModel, SpikeRegressors
    from niworkflows.interfaces.fixes import FixHeaderApplyTransforms as ApplyTransforms
    from niworkflows.interfaces.images import SignalExtraction
    from niworkflows.interfaces.morphology import BinaryDilation, BinarySubtraction
    from niworkflows.interfaces.nibabel import ApplyMask, Binarize
    from niworkflows.interfaces.utility import AddTSVHeader, DictMerge

    from ...interfaces.confounds import aCompCorMasks

    gm_desc = (
        "dilating a GM mask extracted from the FreeSurfer's *aseg* segmentation"
        if freesurfer
        else 'thresholding the corresponding partial volume map at 0.05'
    )

    workflow = Workflow(name=name)
    workflow.__desc__ = f"""\
Several confounding time-series were calculated based on the
*preprocessed PET*: framewise displacement (FD), DVARS and
three region-wise global signals.
FD was computed using two formulations following Power (absolute sum of
relative motions, @power_fd_dvars) and Jenkinson (relative root mean square
displacement between affines, @mcflirt).
FD and DVARS are calculated for each PET run, both using their
implementations in *Nipype* [following the definitions by @power_fd_dvars].
The three global signals are extracted within the CSF, the WM, and
the whole-brain masks.
Additionally, a set of physiological regressors were extracted to
allow for component-based noise correction [*CompCor*, @compcor].
Principal components are estimated after high-pass filtering the
*preprocessed PET* time-series (using a discrete cosine filter with
128s cut-off) for the two *CompCor* variants: temporal (tCompCor)
and anatomical (aCompCor).
tCompCor components are then calculated from the top 2% variable
voxels within the brain mask.
For aCompCor, three probabilistic masks (CSF, WM and combined CSF+WM)
are generated in anatomical space.
The implementation differs from that of Behzadi et al. in that instead
of eroding the masks by 2 pixels on PET space, a mask of pixels that
likely contain a volume fraction of GM is subtracted from the aCompCor masks.
This mask is obtained by {gm_desc}, and it ensures components are not extracted
from voxels containing a minimal fraction of GM.
Finally, these masks are resampled into PET space and binarized by
thresholding at 0.99 (as in the original implementation).
Components are also calculated separately within the WM and CSF masks.
For each CompCor decomposition, the *k* components with the largest singular
values are retained, such that the retained components' time series are
sufficient to explain 50 percent of variance across the nuisance mask (CSF,
WM, combined, or temporal). The remaining components are dropped from
consideration.
The head-motion estimates calculated in the correction step were also
placed within the corresponding confounds file.
The confound time series derived from head motion estimates and global
signals were expanded with the inclusion of temporal derivatives and
quadratic terms for each [@confounds_satterthwaite_2013].
Frames that exceeded a threshold of {regressors_fd_th} mm FD or
{regressors_dvars_th} standardized DVARS were annotated as motion outliers.
Additional nuisance timeseries are calculated by means of principal components
analysis of the signal found within a thin band (*crown*) of voxels around
the edge of the brain, as proposed by [@patriat_improved_2017].
"""
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'pet',
                'pet_mask',
                'petref',
                'motion_xfm',
                't1w_mask',
                't1w_tpms',
                'petref2anat_xfm',
            ]
        ),
        name='inputnode',
    )
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'confounds_file',
                'confounds_metadata',
                'crown_mask',
            ]
        ),
        name='outputnode',
    )

    # Project T1w mask into PET space and merge with PET brainmask
    t1w_mask_tfm = pe.Node(
        ApplyTransforms(interpolation='MultiLabel', invert_transform_flags=[True]),
        name='t1w_mask_tfm',
    )
    union_mask = pe.Node(niu.Function(function=_binary_union), name='union_mask')

    # Create the crown mask
    dilated_mask = pe.Node(BinaryDilation(), name='dilated_mask')
    subtract_mask = pe.Node(BinarySubtraction(), name='subtract_mask')

    # DVARS
    dvars = pe.Node(
        nac.ComputeDVARS(save_nstd=True, save_std=True, remove_zerovariance=True),
        name='dvars',
        mem_gb=mem_gb,
    )

    # Motion parameters
    motion_params = pe.Node(FSLMotionParams(), name='motion_params')

    # Frame displacement
    fdisp = pe.Node(FramewiseDisplacement(), name='fdisp')
    rmsd = pe.Node(FSLRMSDeviation(), name='rmsd')

    # Global and segment regressors
    signals_class_labels = [
        'global_signal',
        'csf',
        'white_matter',
        'csf_wm',
    ]
    get_pet_zooms = pe.Node(niu.Function(function=_get_zooms), name='get_pet_zooms')
    acompcor_masks = pe.Node(aCompCorMasks(), name='acompcor_masks')
    acompcor_tfm = pe.MapNode(
        ApplyTransforms(interpolation='MultiLabel', invert_transform_flags=[True]),
        name='acompcor_tfm',
        iterfield=['input_image'],
    )
    acompcor_bin = pe.MapNode(
        Binarize(thresh_low=0.99),
        name='acompcor_bin',
        iterfield=['in_file'],
    )
    merge_rois = pe.Node(
        niu.Merge(3, ravel_inputs=True), name='merge_rois', run_without_submitting=True
    )
    signals = pe.Node(
        SignalExtraction(class_labels=signals_class_labels), name='signals', mem_gb=mem_gb
    )

    # Arrange confounds
    add_dvars_header = pe.Node(
        AddTSVHeader(columns=['dvars']),
        name='add_dvars_header',
        mem_gb=0.01,
        run_without_submitting=True,
    )
    add_std_dvars_header = pe.Node(
        AddTSVHeader(columns=['std_dvars']),
        name='add_std_dvars_header',
        mem_gb=0.01,
        run_without_submitting=True,
    )
    concat = pe.Node(GatherConfounds(), name='concat', mem_gb=0.01, run_without_submitting=True)

    # Combine all confounds metadata
    mrg_conf_metadata = pe.Node(
        niu.Merge(2), name='merge_confound_metadata', run_without_submitting=True
    )
    # Tissue mean time series
    mrg_conf_metadata.inputs.in1 = {label: {'Method': 'Mean'} for label in signals_class_labels}
    # Movement parameters
    mrg_conf_metadata.inputs.in2 = {
        'trans_x': {'Description': 'Translation along left-right axis.', 'Units': 'mm'},
        'trans_y': {'Description': 'Translation along anterior-posterior axis.', 'Units': 'mm'},
        'trans_z': {'Description': 'Translation along superior-inferior axis.', 'Units': 'mm'},
        'rot_x': {
            'Description': 'Rotation about left-right axis. Also known as "pitch".',
            'Units': 'rad',
        },
        'rot_y': {
            'Description': 'Rotation about anterior-posterior axis. Also known as "roll".',
            'Units': 'rad',
        },
        'rot_z': {
            'Description': 'Rotation about superior-inferior axis. Also known as "yaw".',
            'Units': 'rad',
        },
        'framewise_displacement': {'Units': 'mm'},
    }
    mrg_conf_metadata2 = pe.Node(
        DictMerge(), name='merge_confound_metadata2', run_without_submitting=True
    )

    # Expand model to include derivatives and quadratics
    model_expand = pe.Node(
        ExpandModel(model_formula='(dd1(rps + wm + csf + gsr))^^2 + others'),
        name='model_expansion',
    )

    # Add spike regressors
    spike_regress = pe.Node(
        SpikeRegressors(fd_thresh=regressors_fd_th, dvars_thresh=regressors_dvars_th),
        name='spike_regressors',
    )

    # Generate reportlet (Confound correlation)
    conf_corr_plot = pe.Node(
        ConfoundsCorrelationPlot(reference_column='global_signal', max_dim=20),
        name='conf_corr_plot',
    )
    ds_report_conf_corr = pe.Node(
        DerivativesDataSink(
            desc='confoundcorr', datatype='figures'
        ),
        name='ds_report_conf_corr',
        run_without_submitting=True,
        mem_gb=DEFAULT_MEMORY_MIN_GB,
    )

    def _last(inlist):
        return inlist[-1]

    def _select_cols(table):
        import pandas as pd

        return [
            col
            for col in pd.read_table(table, nrows=2).columns
            if not col.startswith(('a_comp_cor_', 't_comp_cor_', 'std_dvars'))
        ]

    workflow.connect([
        # connect inputnode to each non-anatomical confound node
        (inputnode, dvars, [('pet', 'in_file'),
                            ('pet_mask', 'in_mask')]),
        (inputnode, motion_params, [('motion_xfm', 'xfm_file'),
                                    ('petref', 'petref_file')]),
        (inputnode, rmsd, [('motion_xfm', 'xfm_file'),
                           ('petref', 'petref_file')]),
        (motion_params, fdisp, [('out_file', 'in_file')]),
        # Brain mask
        (inputnode, t1w_mask_tfm, [('t1w_mask', 'input_image'),
                                   ('pet_mask', 'reference_image'),
                                   ('petref2anat_xfm', 'transforms')]),
        (inputnode, union_mask, [('pet_mask', 'mask1')]),
        (t1w_mask_tfm, union_mask, [('output_image', 'mask2')]),
        (union_mask, dilated_mask, [('out', 'in_mask')]),
        (union_mask, subtract_mask, [('out', 'in_subtract')]),
        (dilated_mask, subtract_mask, [('out_mask', 'in_base')]),
        (subtract_mask, outputnode, [('out_mask', 'crown_mask')]),
        # Global signals extraction (constrained by anatomy)
        (inputnode, signals, [('pet', 'in_file')]),
        (inputnode, get_pet_zooms, [('pet', 'in_file')]),
        (inputnode, acompcor_masks, [('t1w_tpms', 'in_vfs')]),
        (get_pet_zooms, acompcor_masks, [('out', 'pet_zooms')]),
        (acompcor_masks, acompcor_tfm, [('out_masks', 'input_image')]),
        (inputnode, acompcor_tfm, [
            ('pet_mask', 'reference_image'),
            ('petref2anat_xfm', 'transforms'),
        ]),
        (acompcor_tfm, acompcor_bin, [('output_image', 'in_file')]),
        (acompcor_bin, merge_rois, [
            (('out_mask', _last), 'in3'),
            (('out_mask', lambda l: l[0]), 'in1'),
            (('out_mask', lambda l: l[1]), 'in2'),
        ]),
        (merge_rois, signals, [('out', 'label_files')]),

        # Collate computed confounds together
        (dvars, add_dvars_header, [('out_nstd', 'in_file')]),
        (dvars, add_std_dvars_header, [('out_std', 'in_file')]),
        (signals, concat, [('out_file', 'signals')]),
        (fdisp, concat, [('out_file', 'fd')]),
        (motion_params, concat, [('out_file', 'motion')]),
        (rmsd, concat, [('out_file', 'rmsd')]),
        (add_dvars_header, concat, [('out_file', 'dvars')]),
        (add_std_dvars_header, concat, [('out_file', 'std_dvars')]),

        # Confounds metadata
        (mrg_conf_metadata, mrg_conf_metadata2, [('out', 'in_dicts')]),

        # Expand the model with derivatives, quadratics, and spikes
        (concat, model_expand, [('confounds_file', 'confounds_file')]),
        (model_expand, spike_regress, [('confounds_file', 'confounds_file')]),

        # Set outputs
        (spike_regress, outputnode, [('confounds_file', 'confounds_file')]),
        (mrg_conf_metadata2, outputnode, [('out_dict', 'confounds_metadata')]),
        (concat, conf_corr_plot, [('confounds_file', 'confounds_file'),
                                  (('confounds_file', _select_cols), 'columns')]),
        (conf_corr_plot, ds_report_conf_corr, [('out_file', 'in_file')]),
    ])  # fmt: skip

    return workflow


def init_carpetplot_wf(
    mem_gb: float, metadata: dict, cifti_output: bool, name: str = 'pet_carpet_wf'
):
    """
    Build a workflow to generate *carpet* plots.

    Resamples the MNI parcellation (ad-hoc parcellation derived from the
    Harvard-Oxford template and others).

    Parameters
    ----------
    mem_gb : :obj:`float`
        Size of PET file in GB - please note that this size
        should be calculated after resamplings that may extend
        the FoV
    metadata : :obj:`dict`
        BIDS metadata for PET file
    name : :obj:`str`
        Name of workflow (default: ``pet_carpet_wf``)

    Inputs
    ------
    pet
        PET image, after the prescribed corrections (STC, HMC and SDC)
        when available.
    pet_mask
        PET series mask
    confounds_file
        TSV of all aggregated confounds
    petref2anat_xfm
        Affine matrix that maps the PET reference space into alignment with
        the anatomical (T1w) space
    std2anat_xfm
        ANTs-compatible affine-and-warp transform file
    cifti_pet
        PET image in CIFTI format, to be used in place of volumetric PET
    crown_mask
        Mask of brain edge voxels

    Outputs
    -------
    out_carpetplot
        Path of the generated SVG file

    """
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow
    from niworkflows.interfaces.fixes import FixHeaderApplyTransforms as ApplyTransforms

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'pet',
                'pet_mask',
                'confounds_file',
                'petref2anat_xfm',
                'std2anat_xfm',
                'cifti_pet',
                'crown_mask',
            ]
        ),
        name='inputnode',
    )

    outputnode = pe.Node(niu.IdentityInterface(fields=['out_carpetplot']), name='outputnode')

    # Carpetplot and confounds plot
    timing_parameters = prepare_timing_parameters(metadata)
    conf_plot = pe.Node(
        PETSummary(
            volume_timing=timing_parameters.get('VolumeTiming'),
            confounds_list=[
                ('global_signal', None, 'GS'),
                ('csf', None, 'CSF'),
                ('white_matter', None, 'WM'),
                ('std_dvars', None, 'DVARS'),
                ('framewise_displacement', 'mm', 'FD'),
            ],
        ),
        name='conf_plot',
        mem_gb=mem_gb,
    )
    ds_report_pet_conf = pe.Node(
        DerivativesDataSink(
            desc='carpetplot', datatype='figures', extension='svg'
        ),
        name='ds_report_pet_conf',
        run_without_submitting=True,
        mem_gb=DEFAULT_MEMORY_MIN_GB,
    )

    parcels = pe.Node(
        niu.Function(
            function=_carpet_parcellation,
            input_names=['segmentation', 'crown_mask', 'nifti'],
            output_names=['out'],
        ),
        name='parcels',
    )
    parcels.inputs.nifti = not cifti_output
    # List transforms
    mrg_xfms = pe.Node(niu.Merge(2), name='mrg_xfms')

    # Warp segmentation into PET space
    resample_parc = pe.Node(
        ApplyTransforms(
            dimension=3,
            input_image=str(
                get_template(
                    'MNI152NLin2009cAsym',
                    resolution=1,
                    desc='carpet',
                    suffix='dseg',
                    extension=['.nii', '.nii.gz'],
                )
            ),
            invert_transform_flags=[True, False],
            interpolation='MultiLabel',
            args='-u int',
        ),
        name='resample_parc',
    )

    workflow = Workflow(name=name)
    if cifti_output:
        workflow.connect(inputnode, 'cifti_pet', conf_plot, 'in_cifti')

    workflow.connect([
        (inputnode, mrg_xfms, [
            ('petref2anat_xfm', 'in1'),
            ('std2anat_xfm', 'in2'),
        ]),
        (inputnode, resample_parc, [('pet_mask', 'reference_image')]),
        (inputnode, parcels, [('crown_mask', 'crown_mask')]),
        (inputnode, conf_plot, [
            ('pet', 'in_nifti'),
            ('confounds_file', 'confounds_file'),
        ]),
        (mrg_xfms, resample_parc, [('out', 'transforms')]),
        (resample_parc, parcels, [('output_image', 'segmentation')]),
        (parcels, conf_plot, [('out', 'in_segm')]),
        (conf_plot, ds_report_pet_conf, [('out_file', 'in_file')]),
        (conf_plot, outputnode, [('out_file', 'out_carpetplot')]),
    ])  # fmt:skip
    return workflow


def _binary_union(mask1, mask2):
    """Generate the union of two masks."""
    from pathlib import Path

    import nibabel as nb
    import numpy as np

    img = nb.load(mask1)
    mskarr1 = np.asanyarray(img.dataobj, dtype=int) > 0
    mskarr2 = np.asanyarray(nb.load(mask2).dataobj, dtype=int) > 0
    out = img.__class__(mskarr1 | mskarr2, img.affine, img.header)
    out.set_data_dtype('uint8')
    out_name = Path('mask_union.nii.gz').absolute()
    out.to_filename(out_name)
    return str(out_name)


def _carpet_parcellation(segmentation, crown_mask, nifti=False):
    """Generate a segmentation for carpet plot visualization."""
    from pathlib import Path

    import nibabel as nb
    import numpy as np

    img = nb.load(segmentation)

    lut = np.zeros((256,), dtype='uint8')
    lut[100:201] = 1 if nifti else 0  # Ctx GM
    lut[30:99] = 2 if nifti else 0  # dGM
    lut[1:11] = 3 if nifti else 1  # WM+CSF
    lut[255] = 5 if nifti else 0  # Cerebellum
    # Apply lookup table
    seg = lut[np.uint16(img.dataobj)]
    seg[np.bool_(nb.load(crown_mask).dataobj)] = 6 if nifti else 2

    outimg = img.__class__(seg.astype('uint8'), img.affine, img.header)
    outimg.set_data_dtype('uint8')
    out_file = Path('segments.nii.gz').absolute()
    outimg.to_filename(out_file)
    return str(out_file)


def _get_zooms(in_file):
    import nibabel as nb

    return tuple(nb.load(in_file).header.get_zooms()[:3])
