# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""The freesurfer module provides basic functions for interfacing with
   freesurfer tools.
"""
from __future__ import print_function, division, unicode_literals, absolute_import

import os

from nipype.utils.filemanip import fname_presuffix, split_filename
from nipype.interfaces.base import (
    TraitedSpec,
    File,
    traits,
    InputMultiPath,
    OutputMultiPath,
    Directory,
    isdefined,
)
from nipype.interfaces.freesurfer.base import FSCommand, FSTraitedSpec
from nipype.interfaces.freesurfer.utils import copy2subjdir

__docformat__ = "restructuredtext"


class SegStatsInputSpec(FSTraitedSpec):
    _xor_inputs = ("segmentation_file", "annot", "surf_label")
    segmentation_file = File(
        exists=True,
        argstr="--seg %s",
        xor=_xor_inputs,
        mandatory=True,
        desc="segmentation volume path",
    )
    annot = traits.Tuple(
        traits.Str,
        traits.Enum("lh", "rh"),
        traits.Str,
        argstr="--annot %s %s %s",
        xor=_xor_inputs,
        mandatory=True,
        desc="subject hemi parc : use surface parcellation",
    )
    surf_label = traits.Tuple(
        traits.Str,
        traits.Enum("lh", "rh"),
        traits.Str,
        argstr="--slabel %s %s %s",
        xor=_xor_inputs,
        mandatory=True,
        desc="subject hemi label : use surface label",
    )
    summary_file = File(
        argstr="--sum %s",
        genfile=True,
        position=-1,
        desc="Segmentation stats summary table file",
    )
    partial_volume_file = File(
        exists=True, argstr="--pv %s", desc="Compensate for partial voluming"
    )
    in_file = File(
        exists=True,
        argstr="--i %s",
        desc="Use the segmentation to report stats on this volume",
    )
    frame = traits.Int(
        argstr="--frame %d", desc="Report stats on nth frame of input volume"
    )
    multiply = traits.Float(argstr="--mul %f", desc="multiply input by val")
    calc_snr = traits.Bool(
        argstr="--snr", desc="save mean/std as extra column in output table"
    )
    calc_power = traits.Enum(
        "sqr",
        "sqrt",
        argstr="--%s",
        desc="Compute either the sqr or the sqrt of the input",
    )
    _ctab_inputs = ("color_table_file", "default_color_table", "gca_color_table")
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
    gca_color_table = File(
        exists=True,
        argstr="--ctab-gca %s",
        xor=_ctab_inputs,
        desc="get color table from GCA (CMA)",
    )
    segment_id = traits.List(
        argstr="--id %s...", desc="Manually specify segmentation ids"
    )
    exclude_id = traits.Int(argstr="--excludeid %d", desc="Exclude seg id from report")
    exclude_ctx_gm_wm = traits.Bool(
        argstr="--excl-ctxgmwm", desc="exclude cortical gray and white matter"
    )
    wm_vol_from_surf = traits.Bool(
        argstr="--surf-wm-vol", desc="Compute wm volume from surf"
    )
    cortex_vol_from_surf = traits.Bool(
        argstr="--surf-ctx-vol", desc="Compute cortex volume from surf"
    )
    non_empty_only = traits.Bool(
        argstr="--nonempty", desc="Only report nonempty segmentations"
    )
    ctab_out_file = File(
        argstr="--ctab-out %s",
        desc="Write color table used to segmentation file",
        genfile=True,
        position=-2,
    )
    empty = traits.Bool(
        argstr="--empty", desc="Report on segmentations listed in the color table"
    )
    mask_file = File(
        exists=True, argstr="--mask %s", desc="Mask volume (same size as seg"
    )
    mask_thresh = traits.Float(
        argstr="--maskthresh %f", desc="binarize mask with this threshold <0.5>"
    )
    mask_sign = traits.Enum(
        "abs",
        "pos",
        "neg",
        "--masksign %s",
        desc="Sign for mask threshold: pos, neg, or abs",
    )
    mask_frame = traits.Int(
        "--maskframe %d",
        requires=["mask_file"],
        desc="Mask with this (0 based) frame of the mask volume",
    )
    mask_invert = traits.Bool(
        argstr="--maskinvert", desc="Invert binarized mask volume"
    )
    mask_erode = traits.Int(argstr="--maskerode %d", desc="Erode mask by some amount")
    brain_vol = traits.Enum(
        "brain-vol-from-seg",
        "brainmask",
        argstr="--%s",
        desc="Compute brain volume either with ``brainmask`` or ``brain-vol-from-seg``",
    )
    brainmask_file = File(
        argstr="--brainmask %s",
        exists=True,
        desc="Load brain mask and compute the volume of the brain as the non-zero voxels in this volume",
    )
    etiv = traits.Bool(argstr="--etiv", desc="Compute ICV from talairach transform")
    etiv_only = traits.Enum(
        "etiv",
        "old-etiv",
        "--%s-only",
        desc="Compute etiv and exit.  Use ``etiv`` or ``old-etiv``",
    )
    avgwf_txt_file = traits.Either(
        traits.Bool,
        File,
        argstr="--avgwf %s",
        desc="Save average waveform into file (bool or filename)",
    )
    avgwf_file = traits.Either(
        traits.Bool,
        File,
        argstr="--avgwfvol %s",
        desc="Save as binary volume (bool or filename)",
    )
    sf_avg_file = traits.Either(
        traits.Bool, File, argstr="--sfavg %s", desc="Save mean across space and time"
    )
    vox = traits.List(
        traits.Int,
        argstr="--vox %s",
        desc="Replace seg with all 0s except at C R S (three int inputs)",
    )
    supratent = traits.Bool(argstr="--supratent", desc="Undocumented input flag")
    subcort_gm = traits.Bool(
        argstr="--subcortgray", desc="Compute volume of subcortical gray matter"
    )
    total_gray = traits.Bool(
        argstr="--totalgray", desc="Compute volume of total gray matter"
    )
    euler = traits.Bool(
        argstr="--euler",
        desc="Write out number of defect holes in orig.nofix based on the euler number",
    )
    in_intensity = File(
        argstr="--in %s --in-intensity-name %s", desc="Undocumented input norm.mgz file"
    )
    intensity_units = traits.Enum(
        "MR",
        argstr="--in-intensity-units %s",
        requires=["in_intensity"],
        desc="Intensity units",
    )


class SegStatsOutputSpec(TraitedSpec):
    summary_file = File(exists=True, desc="Segmentation summary statistics table")
    avgwf_txt_file = File(
        desc="Text file with functional statistics averaged over segs"
    )
    avgwf_file = File(desc="Volume with functional statistics averaged over segs")
    sf_avg_file = File(
        desc="Text file with func statistics averaged over segs and framss"
    )
    ctab_out_file = File(
        exists=True, desc="Color table used to generate segmentation file"
    )


class SegStats(FSCommand):
    """Use FreeSurfer mri_segstats for ROI analysis

    Examples
    --------

    >>> import nipype.interfaces.freesurfer as fs
    >>> ss = fs.SegStats()
    >>> ss.inputs.annot = ('PWS04', 'lh', 'aparc')
    >>> ss.inputs.in_file = 'functional.nii'
    >>> ss.inputs.subjects_dir = '.'
    >>> ss.inputs.avgwf_txt_file = 'avgwf.txt'
    >>> ss.inputs.summary_file = 'summary.stats'
    >>> ss.cmdline
    'mri_segstats --annot PWS04 lh aparc --avgwf ./avgwf.txt --i functional.nii --sum ./summary.stats'

    """

    _cmd = "mri_segstats"
    input_spec = SegStatsInputSpec
    output_spec = SegStatsOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        if isdefined(self.inputs.summary_file):
            outputs["summary_file"] = os.path.abspath(self.inputs.summary_file)
        else:
            outputs["summary_file"] = os.path.join(os.getcwd(), "summary.stats")

        if isdefined(self.inputs.ctab_out_file):
            outputs["ctab_out_file"] = os.path.abspath(self.inputs.ctab_out_file)
        else:
            outputs["ctab_out_file"] = os.path.join(os.getcwd(), "ctab_out.ctab")

        suffices = dict(
            avgwf_txt_file="_avgwf.txt",
            avgwf_file="_avgwf.nii.gz",
            sf_avg_file="sfavg.txt",
        )
        if isdefined(self.inputs.segmentation_file):
            _, src = os.path.split(self.inputs.segmentation_file)
        if isdefined(self.inputs.annot):
            src = "_".join(self.inputs.annot)
        if isdefined(self.inputs.surf_label):
            src = "_".join(self.inputs.surf_label)
        for name, suffix in list(suffices.items()):
            value = getattr(self.inputs, name)
            if isdefined(value):
                if isinstance(value, bool):
                    outputs[name] = fname_presuffix(
                        src, suffix=suffix, newpath=os.getcwd(), use_ext=False
                    )
                else:
                    outputs[name] = os.path.abspath(value)
        return outputs

    def _format_arg(self, name, spec, value):
        if name in ("summary_file", "ctab_out_file", "avgwf_txt_file"):
            if not isinstance(value, bool):
                if not os.path.isabs(value):
                    value = os.path.join(".", value)
        if name in ["avgwf_txt_file", "avgwf_file", "sf_avg_file"]:
            if isinstance(value, bool):
                fname = self._list_outputs()[name]
            else:
                fname = value
            return spec.argstr % fname
        elif name == "in_intensity":
            intensity_name = os.path.basename(self.inputs.in_intensity).replace(
                ".mgz", ""
            )
            return spec.argstr % (value, intensity_name)
        return super(SegStats, self)._format_arg(name, spec, value)

    def _gen_filename(self, name):
        if name == "summary_file" or name == "ctab_out_file":
            return self._list_outputs()[name]
        return None
