import nibabel as nb
import numpy as np
import pandas as pd
from nipype.interfaces.base import (
    BaseInterface,
    BaseInterfaceInputSpec,
    TraitedSpec,
    File,
    traits,
    InputMultiPath,
    Directory,
    isdefined,
)
import os

from nipype.interfaces.freesurfer.base import FSCommand, FSTraitedSpec


class Binarise4DSegmentationInputSpec(BaseInterfaceInputSpec):
    dseg_file = File(exists=True, mandatory=True, desc="Input segmentation file (_dseg.nii.gz)")
    out_file = File("binarised_4d.nii.gz", usedefault=True, desc="Output 4D binary segmentation")


class Binarise4DSegmentationOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="Output 4D binary segmentation file")
    label_list = traits.List(traits.Int, desc="List of labels corresponding to the segmentation regions")


class Binarise4DSegmentation(BaseInterface):
    input_spec = Binarise4DSegmentationInputSpec
    output_spec = Binarise4DSegmentationOutputSpec

    def _run_interface(self, runtime):
        dseg_img = nb.load(self.inputs.dseg_file)
        dseg_data = dseg_img.get_fdata().astype(np.int32)

        region_labels = np.unique(dseg_data)
        region_labels = region_labels[region_labels != 0]  # exclude background

        binarised_4d = np.zeros(dseg_data.shape + (len(region_labels),), dtype=np.uint8)

        for idx, label in enumerate(region_labels):
            binarised_4d[..., idx] = (dseg_data == label).astype(np.uint8)

        new_img = nb.Nifti1Image(binarised_4d, affine=dseg_img.affine, header=dseg_img.header)
        nb.save(new_img, os.path.abspath(self.inputs.out_file))

        self._label_list = region_labels.tolist()

        return runtime

    def _list_outputs(self):
        return {
            "out_file": os.path.abspath(self.inputs.out_file),
            "label_list": self._label_list
        }


class StackTissueProbabilityMapsInputSpec(BaseInterfaceInputSpec):
    t1w_tpms = traits.List(File(exists=True), mandatory=True, desc="List of T1w tissue probability maps")
    out_file = File("stacked_probseg.nii.gz", usedefault=True, desc="Output stacked 4D probability map")


class StackTissueProbabilityMapsOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="Output stacked 4D probability map")


class StackTissueProbabilityMaps(BaseInterface):
    input_spec = StackTissueProbabilityMapsInputSpec
    output_spec = StackTissueProbabilityMapsOutputSpec

    def _run_interface(self, runtime):
        imgs = [nb.load(f).get_fdata() for f in self.inputs.t1w_tpms]
        affine = nb.load(self.inputs.t1w_tpms[0]).affine

        stacked_img = np.stack(imgs, axis=-1)

        new_img = nb.Nifti1Image(stacked_img, affine=affine)
        nb.save(new_img, os.path.abspath(self.inputs.out_file))

        return runtime

    def _list_outputs(self):
        return {"out_file": os.path.abspath(self.inputs.out_file)}


class CSVtoNiftiInputSpec(BaseInterfaceInputSpec):
    csv_file = File(exists=True, mandatory=True, desc="Input CSV file with region means")
    reference_nifti = File(exists=True, mandatory=True, desc="Reference NIfTI file for spatial information")
    label_list = traits.List(traits.Int, mandatory=True, desc="List of labels corresponding to regions")
    out_file = File("output_from_csv.nii.gz", usedefault=True, desc="Output NIfTI file")


class CSVtoNiftiOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="Output NIfTI image file")


class CSVtoNifti(BaseInterface):
    input_spec = CSVtoNiftiInputSpec
    output_spec = CSVtoNiftiOutputSpec

    def _run_interface(self, runtime):
        import pandas as pd

        csv_data = pd.read_csv(self.inputs.csv_file, sep='\t')
        reference_img = nb.load(self.inputs.reference_nifti)
        reference_data = reference_img.get_fdata().astype(np.int32)

        output_data = np.zeros(reference_data.shape, dtype=np.float32)

        for idx, label in enumerate(self.inputs.label_list):
            mean_value = csv_data.iloc[idx]['MEAN']
            output_data[reference_data == label] = mean_value

        output_img = nb.Nifti1Image(output_data, affine=reference_img.affine, header=reference_img.header)
        nb.save(output_img, os.path.abspath(self.inputs.out_file))

        return runtime

    def _list_outputs(self):
        return {"out_file": os.path.abspath(self.inputs.out_file)}
    

class GTMPVCInputSpec(FSTraitedSpec):
    in_file = File(
        exists=True,
        argstr="--i %s",
        mandatory=True,
        copyfile=False,
        desc="input volume - source data to pvc",
    )

    frame = traits.Int(
        argstr="--frame %i", desc="only process 0-based frame F from inputvol"
    )

    psf = traits.Float(argstr="--psf %f", desc="scanner PSF FWHM in mm")

    segmentation = File(
        argstr="--seg %s",
        exists=True,
        mandatory=True,
        desc="segfile : anatomical segmentation to define regions for GTM",
    )

    _reg_xor = ["reg_file", "regheader", "reg_identity"]
    reg_file = File(
        exists=True,
        argstr="--reg %s",
        mandatory=True,
        desc="LTA registration file that maps PET to anatomical",
        xor=_reg_xor,
    )

    regheader = traits.Bool(
        argstr="--regheader",
        mandatory=True,
        desc="assume input and seg share scanner space",
        xor=_reg_xor,
    )

    reg_identity = traits.Bool(
        argstr="--reg-identity",
        mandatory=True,
        desc="assume that input is in anatomical space",
        xor=_reg_xor,
    )

    pvc_dir = traits.Str(argstr="--o %s", desc="save outputs to dir", genfile=True)

    mask_file = File(
        exists=True,
        argstr="--mask %s",
        desc="ignore areas outside of the mask (in input vol space)",
    )

    auto_mask = traits.Tuple(
        traits.Float,
        traits.Float,
        argstr="--auto-mask %f %f",
        desc="FWHM thresh : automatically compute mask",
    )

    no_reduce_fov = traits.Bool(
        argstr="--no-reduce-fov", desc="do not reduce FoV to encompass mask"
    )

    reduce_fox_eqodd = traits.Bool(
        argstr="--reduce-fox-eqodd",
        desc="reduce FoV to encompass mask but force nc=nr and ns to be odd",
    )

    contrast = InputMultiPath(
        File(exists=True), argstr="--C %s...", desc="contrast file"
    )

    default_seg_merge = traits.Bool(
        argstr="--default-seg-merge", desc="default schema for merging ROIs"
    )

    merge_hypos = traits.Bool(
        argstr="--merge-hypos", desc="merge left and right hypointensites into to ROI"
    )

    merge_cblum_wm_gyri = traits.Bool(
        argstr="--merge-cblum-wm-gyri",
        desc="cerebellum WM gyri back into cerebellum WM",
    )

    tt_reduce = traits.Bool(
        argstr="--tt-reduce", desc="reduce segmentation to that of a tissue type"
    )

    replace = traits.Tuple(
        traits.Int,
        traits.Int,
        argstr="--replace %i %i",
        desc="Id1 Id2 : replace seg Id1 with seg Id2",
    )

    rescale = traits.List(
        argstr="--rescale %s...",
        desc="Id1 <Id2...>  : specify reference region(s) used to rescale (default is pons)",
    )

    no_rescale = traits.Bool(
        argstr="--no-rescale",
        desc="do not global rescale such that mean of reference region is scaleref",
    )

    scale_refval = traits.Float(
        argstr="--scale-refval %f",
        desc="refval : scale such that mean in reference region is refval",
    )

    _ctab_inputs = ("color_table_file", "default_color_table")
    color_table_file = File(
        exists=True,
        argstr="--ctab %s",
        xor=_ctab_inputs,
        desc="color table file with seg id names",
    )

    default_color_table = traits.Bool(
        argstr="--ctab-default",
        xor=_ctab_inputs,
        desc="use $FREESURFER_HOME/FreeSurferColorLUT.txt",
    )

    tt_update = traits.Bool(
        argstr="--tt-update",
        desc="changes tissue type of VentralDC, BrainStem, and Pons to be SubcortGM",
    )

    lat = traits.Bool(argstr="--lat", desc="lateralize tissue types")

    no_tfe = traits.Bool(
        argstr="--no-tfe",
        desc="do not correct for tissue fraction effect (with --psf 0 turns off PVC entirely)",
    )

    no_pvc = traits.Bool(
        argstr="--no-pvc",
        desc="turns off PVC entirely (both PSF and TFE)",
    )

    tissue_fraction_resolution = traits.Float(
        argstr="--segpvfres %f",
        desc="set the tissue fraction resolution parameter (def is 0.5)",
    )

    rbv = traits.Bool(
        argstr="--rbv",
        requires=["subjects_dir"],
        desc="perform Region-based Voxelwise (RBV) PVC",
    )

    rbv_res = traits.Float(
        argstr="--rbv-res %f",
        desc="voxsize : set RBV voxel resolution (good for when standard res takes too much memory)",
    )

    mg = traits.Tuple(
        traits.Float,
        traits.List(traits.String),
        argstr="--mg %g %s",
        desc="gmthresh RefId1 RefId2 ...: perform Mueller-Gaertner PVC, gmthresh is min gm pvf bet 0 and 1",
    )

    mg_ref_cerebral_wm = traits.Bool(
        argstr="--mg-ref-cerebral-wm", desc=" set MG RefIds to 2 and 41"
    )

    mg_ref_lobes_wm = traits.Bool(
        argstr="--mg-ref-lobes-wm",
        desc="set MG RefIds to those for lobes when using wm subseg",
    )

    mgx = traits.Float(
        argstr="--mgx %f",
        desc="gmxthresh : GLM-based Mueller-Gaertner PVC, gmxthresh is min gm pvf bet 0 and 1",
    )

    km_ref = traits.List(
        argstr="--km-ref %s...",
        desc="RefId1 RefId2 ... : compute reference TAC for KM as mean of given RefIds",
    )

    km_hb = traits.List(
        argstr="--km-hb %s...",
        desc="RefId1 RefId2 ... : compute HiBinding TAC for KM as mean of given RefIds",
    )

    steady_state_params = traits.Tuple(
        traits.Float,
        traits.Float,
        traits.Float,
        argstr="--ss %f %f %f",
        desc="bpc scale dcf : steady-state analysis spec blood plasma concentration, unit scale and decay correction factor. You must also spec --km-ref. Turns off rescaling",
    )

    X = traits.Bool(
        argstr="--X", desc="save X matrix in matlab4 format as X.mat (it will be big)"
    )

    y = traits.Bool(argstr="--y", desc="save y matrix in matlab4 format as y.mat")

    beta = traits.Bool(
        argstr="--beta", desc="save beta matrix in matlab4 format as beta.mat"
    )

    X0 = traits.Bool(
        argstr="--X0",
        desc="save X0 matrix in matlab4 format as X0.mat (it will be big)",
    )

    save_input = traits.Bool(
        argstr="--save-input", desc="saves rescaled input as input.rescaled.nii.gz"
    )

    save_eres = traits.Bool(argstr="--save-eres", desc="saves residual error")

    save_yhat = traits.Bool(
        argstr="--save-yhat",
        xor=["save_yhat_with_noise"],
        desc="save signal estimate (yhat) smoothed with the PSF",
    )

    save_yhat_with_noise = traits.Tuple(
        traits.Int,
        traits.Int,
        argstr="--save-yhat-with-noise %i %i",
        xor=["save_yhat"],
        desc="seed nreps : save signal estimate (yhat) with noise",
    )

    save_yhat_full_fov = traits.Bool(
        argstr="--save-yhat-full-fov", desc="save signal estimate (yhat)"
    )

    save_yhat0 = traits.Bool(argstr="--save-yhat0", desc="save signal estimate (yhat)")

    optimization_schema = traits.Enum(
        "3D",
        "2D",
        "1D",
        "3D_MB",
        "2D_MB",
        "1D_MB",
        "MBZ",
        "MB3",
        argstr="--opt %s",
        desc="opt : optimization schema for applying adaptive GTM",
    )

    opt_tol = traits.Tuple(
        traits.Int,
        traits.Float,
        traits.Float,
        argstr="--opt-tol %i %f %f",
        desc="n_iters_max ftol lin_min_tol : optimization parameters for adaptive gtm using fminsearch",
    )

    opt_brain = traits.Bool(argstr="--opt-brain", desc="apply adaptive GTM")

    opt_seg_merge = traits.Bool(
        argstr="--opt-seg-merge",
        desc="optimal schema for merging ROIs when applying adaptive GTM",
    )

    num_threads = traits.Int(
        argstr="--threads %i", desc="threads : number of threads to use"
    )

    psf_col = traits.Float(
        argstr="--psf-col %f", desc="xFWHM : full-width-half-maximum in the x-direction"
    )

    psf_row = traits.Float(
        argstr="--psf-row %f", desc="yFWHM : full-width-half-maximum in the y-direction"
    )

    psf_slice = traits.Float(
        argstr="--psf-slice %f",
        desc="zFWHM : full-width-half-maximum in the z-direction",
    )


class GTMPVCOutputSpec(TraitedSpec):
    pvc_dir = Directory(desc="output directory")
    ref_file = File(desc="Reference TAC in .dat")
    hb_nifti = File(desc="High-binding TAC in nifti")
    hb_dat = File(desc="High-binding TAC in .dat")
    nopvc_file = File(desc="TACs for all regions with no PVC")
    gtm_file = File(desc="TACs for all regions with GTM PVC")
    gtm_stats = File(desc="Statistics for the GTM PVC")
    input_file = File(desc="4D PET file in native volume space")
    tissue_fraction = File(
        desc="Tissue fraction map in native volume space"
    )
    reg_pet2anat = File(desc="Registration file to go from PET to anat")
    reg_anat2pet = File(desc="Registration file to go from anat to PET")
    reg_rbvpet2anat = File(
        desc="Registration file to go from RBV corrected PET to anat"
    )
    reg_anat2rbvpet = File(
        desc="Registration file to go from anat to RBV corrected PET"
    )
    mgx_ctxgm = File(
        desc="Cortical GM voxel-wise values corrected using the extended Muller-Gartner method",
    )
    mgx_subctxgm = File(
        desc="Subcortical GM voxel-wise values corrected using the extended Muller-Gartner method",
    )
    mgx_gm = File(
        desc="All GM voxel-wise values corrected using the extended Muller-Gartner method",
    )
    mg = File(
        desc="All voxel-wise values corrected using the Muller-Gartner method",
    )
    rbv = File(desc="All GM voxel-wise values corrected using the RBV method")
    opt_params = File(
        desc="Optimal parameter estimates for the FWHM using adaptive GTM"
    )
    yhat0 = File(desc="4D PET file of signal estimate (yhat) after PVC (unsmoothed)")
    yhat = File(
        desc="4D PET file of signal estimate (yhat) after PVC (smoothed with PSF)",
    )
    yhat_full_fov = File(
        desc="4D PET file with full FOV of signal estimate (yhat) after PVC (smoothed with PSF)",
    )
    yhat_with_noise = File(
        desc="4D PET file with full FOV of signal estimate (yhat) with noise after PVC (smoothed with PSF)",
    )


class GTMPVC(FSCommand):
    """create an anatomical segmentation for the geometric transfer matrix (GTM).

    Examples
    --------
    >>> gtmpvc = GTMPVC()
    >>> gtmpvc.inputs.in_file = 'sub-01_ses-baseline_pet.nii.gz'
    >>> gtmpvc.inputs.segmentation = 'gtmseg.mgz'
    >>> gtmpvc.inputs.reg_file = 'sub-01_ses-baseline_pet_mean_reg.lta'
    >>> gtmpvc.inputs.pvc_dir = 'pvc'
    >>> gtmpvc.inputs.psf = 4
    >>> gtmpvc.inputs.default_seg_merge = True
    >>> gtmpvc.inputs.auto_mask = (1, 0.1)
    >>> gtmpvc.inputs.km_ref = ['8 47']
    >>> gtmpvc.inputs.km_hb = ['11 12 50 51']
    >>> gtmpvc.inputs.no_rescale = True
    >>> gtmpvc.inputs.save_input = True
    >>> gtmpvc.cmdline  # doctest: +NORMALIZE_WHITESPACE
    'mri_gtmpvc --auto-mask 1.000000 0.100000 --default-seg-merge \
    --i sub-01_ses-baseline_pet.nii.gz --km-hb 11 12 50 51 --km-ref 8 47 --no-rescale \
    --psf 4.000000 --o pvc --reg sub-01_ses-baseline_pet_mean_reg.lta --save-input \
    --seg gtmseg.mgz'

    >>> gtmpvc = GTMPVC()
    >>> gtmpvc.inputs.in_file = 'sub-01_ses-baseline_pet.nii.gz'
    >>> gtmpvc.inputs.segmentation = 'gtmseg.mgz'
    >>> gtmpvc.inputs.regheader = True
    >>> gtmpvc.inputs.pvc_dir = 'pvc'
    >>> gtmpvc.inputs.mg = (0.5, ["ROI1", "ROI2"])
    >>> gtmpvc.cmdline  # doctest: +NORMALIZE_WHITESPACE
    'mri_gtmpvc --i sub-01_ses-baseline_pet.nii.gz --mg 0.5 ROI1 ROI2 --o pvc --regheader --seg gtmseg.mgz'
    """

    _cmd = "mri_gtmpvc"
    input_spec = GTMPVCInputSpec
    output_spec = GTMPVCOutputSpec

    def _format_arg(self, name, spec, val):
        # Values taken from
        # https://github.com/freesurfer/freesurfer/blob/fs-7.2/mri_gtmpvc/mri_gtmpvc.cpp#L115-L122
        if name == "optimization_schema":
            return (
                spec.argstr
                % {
                    "3D": 1,
                    "2D": 2,
                    "1D": 3,
                    "3D_MB": 4,
                    "2D_MB": 5,
                    "1D_MB": 6,
                    "MBZ": 7,
                    "MB3": 8,
                }[val]
            )
        if name == "mg":
            return spec.argstr % (val[0], " ".join(val[1]))
        return super(GTMPVC, self)._format_arg(name, spec, val)

    def _list_outputs(self):
        outputs = self.output_spec().get()
        # Get the top-level output directory
        if not isdefined(self.inputs.pvc_dir):
            pvcdir = os.getcwd()
        else:
            pvcdir = os.path.abspath(self.inputs.pvc_dir)
        outputs["pvc_dir"] = pvcdir

        # Assign the output files that always get created
        outputs["ref_file"] = os.path.join(pvcdir, "km.ref.tac.dat")
        outputs["hb_nifti"] = os.path.join(pvcdir, "km.hb.tac.nii.gz")
        outputs["hb_dat"] = os.path.join(pvcdir, "km.hb.tac.dat")
        outputs["nopvc_file"] = os.path.join(pvcdir, "nopvc.nii.gz")
        outputs["gtm_file"] = os.path.join(pvcdir, "gtm.nii.gz")
        outputs["gtm_stats"] = os.path.join(pvcdir, "gtm.stats.dat")
        outputs["reg_pet2anat"] = os.path.join(pvcdir, "aux", "bbpet2anat.lta")
        outputs["reg_anat2pet"] = os.path.join(pvcdir, "aux", "anat2bbpet.lta")
        outputs["tissue_fraction"] = os.path.join(pvcdir, "aux", "tissue.fraction.nii.gz")

        # Assign the conditional outputs
        if self.inputs.save_input:
            outputs["input_file"] = os.path.join(pvcdir, "input.nii.gz")
        if self.inputs.save_yhat0:
            outputs["yhat0"] = os.path.join(pvcdir, "yhat0.nii.gz")
        if self.inputs.save_yhat:
            outputs["yhat"] = os.path.join(pvcdir, "yhat.nii.gz")
        if self.inputs.save_yhat_full_fov:
            outputs["yhat_full_fov"] = os.path.join(pvcdir, "yhat.fullfov.nii.gz")
        if self.inputs.save_yhat_with_noise:
            outputs["yhat_with_noise"] = os.path.join(pvcdir, "yhat.nii.gz")
        if self.inputs.mgx:
            outputs["mgx_ctxgm"] = os.path.join(pvcdir, "mgx.ctxgm.nii.gz")
            outputs["mgx_subctxgm"] = os.path.join(pvcdir, "mgx.subctxgm.nii.gz")
            outputs["mgx_gm"] = os.path.join(pvcdir, "mgx.gm.nii.gz")
        if self.inputs.mg:
            outputs["mg"] = os.path.join(pvcdir, "mg.nii.gz")
        if self.inputs.rbv:
            outputs["rbv"] = os.path.join(pvcdir, "rbv.nii.gz")
            outputs["reg_rbvpet2anat"] = os.path.join(pvcdir, "aux", "rbv2anat.lta")
            outputs["reg_anat2rbvpet"] = os.path.join(pvcdir, "aux", "anat2rbv.lta")
        if self.inputs.optimization_schema:
            outputs["opt_params"] = os.path.join(pvcdir, "aux", "opt.params.dat")

        return outputs


class GTMStatsTo4DNiftiInputSpec(BaseInterfaceInputSpec):
    gtm_file = File(exists=True, mandatory=True, desc='Input GTM NIfTI file from PETSurfer')
    segmentation = File(exists=True, mandatory=True, desc='Input segmentation NIfTI file')
    gtm_stats = File(exists=True, mandatory=True, desc='GTM statistics file (gtm.stats.dat)')
    out_file = traits.Str('gtm_4d.nii.gz', usedefault=True, desc='Output filename')


class GTMStatsTo4DNiftiOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='Output 4D NIfTI file')


class GTMStatsTo4DNifti(BaseInterface):
    input_spec = GTMStatsTo4DNiftiInputSpec
    output_spec = GTMStatsTo4DNiftiOutputSpec

    def _run_interface(self, runtime):
        # Load segmentation
        seg_img = nb.load(self.inputs.segmentation)
        seg_data = seg_img.get_fdata().astype(int)

        # Load GTM data
        gtm_img = nb.load(self.inputs.gtm_file)
        gtm_data = gtm_img.get_fdata()
        
        # Load GTM stats
        gtm_stats = pd.read_csv(
            self.inputs.gtm_stats, delim_whitespace=True, header=None, usecols=[1], names=['index']
        )
        gtm_indices = gtm_stats['index'].values

        n_frames = gtm_data.shape[-1]
        shape_4d = seg_data.shape + (n_frames,)

        output_4d = np.zeros(shape_4d, dtype=np.float32)

        # Map values to regions
        for i, idx in enumerate(gtm_indices):
            mask = seg_data == idx
            output_4d[mask, :] = gtm_data[i, 0, 0, :]

        out_img = nb.Nifti1Image(output_4d, affine=seg_img.affine)
        nb.save(out_img, os.path.abspath(self.inputs.out_file))

        return runtime

    def _list_outputs(self):
        return {"out_file": os.path.abspath(self.inputs.out_file)}
