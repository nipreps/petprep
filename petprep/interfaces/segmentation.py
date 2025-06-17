from __future__ import annotations, print_function, division, unicode_literals, absolute_import

from pathlib import Path
import os
import subprocess

from nipype.utils.filemanip import fname_presuffix, split_filename
from nipype.interfaces.base import (
    BaseInterface,
    BaseInterfaceInputSpec,
    TraitedSpec,
    CommandLine,
    CommandLineInputSpec,
    SimpleInterface,
    File,
    Directory,
    traits,
    InputMultiObject,
    isdefined,
)

from nipype.interfaces.freesurfer.base import FSCommand, FSTraitedSpec
from nipype.interfaces.freesurfer.utils import copy2subjdir


class SegmentBSInputSpec(BaseInterfaceInputSpec):
    subjects_dir = Directory(exists=True, mandatory=True, desc='FreeSurfer subjects directory')
    subject_id = traits.Str(mandatory=True, desc='Subject identifier')


class SegmentBSOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='Brainstem segmentation in anatomical space')
    out_fsvox_file = File(exists=True, desc='Brainstem segmentation in FS voxel space')
    volumes_file = File(exists=True, desc='Brainstem volumes table')
    stderr = File(desc='Standard error output file', exists=True)


class SegmentBS(SimpleInterface):
    """Run ``segmentBS.sh`` if outputs are missing."""

    input_spec = SegmentBSInputSpec
    output_spec = SegmentBSOutputSpec

    def _run_interface(self, runtime):
        subj_dir = Path(self.inputs.subjects_dir) / self.inputs.subject_id / 'mri'
        out_file = subj_dir / 'brainstemSsLabels.v13.mgz'
        out_fsvox = subj_dir / 'brainstemSsLabels.v13.FSvoxelSpace.mgz'
        volumes = subj_dir / 'brainstemSsVolumes.v13.txt'

        if not (out_file.exists() and out_fsvox.exists() and volumes.exists()):
            cmd = [
                'segmentBS.sh',
                self.inputs.subject_id,
                str(self.inputs.subjects_dir),
            ]
            self._results['stdout'], self._results['stderr'] = self._run_command(cmd)
        else:
            runtime.returncode = 0

        self._results.update(
            {
                'out_file': str(out_file),
                'out_fsvox_file': str(out_fsvox),
                'volumes_file': str(volumes),
            }
        )
        return runtime

    def _run_command(self, cmd):
        import subprocess

        proc = subprocess.run(cmd, capture_output=True, text=True)
        return proc.stdout, proc.stderr


class MRISclimbicSegInputSpec(CommandLineInputSpec):
    in_file = File(desc="T1-w image(s) to segment.", exists=True, argstr="--i %s")
    out_file = File(desc="Segmentation output.", argstr="--o %s")
    subjects = InputMultiObject(
        traits.Str,
        desc="Process a series of freesurfer recon-all subjects.",
        argstr="--s %s",
    )
    sd = Directory(desc="Set the subjects directory.", argstr="--sd %s")
    conform = traits.Bool(desc="Resample input to 1mm-iso.", argstr="--conform")
    etiv = traits.Bool(desc="Include eTIV in volume stats.", argstr="--etiv")
    tal = File(
        desc="Alternative talairach xfm transform for estimating TIV.",
        argstr="--tal %s",
    )
    write_posteriors = traits.Bool(
        desc="Save the label posteriors.", argstr="--write_posteriors"
    )
    write_volumes = traits.Bool(
        desc="Save label volume stats.", argstr="--write_volumes"
    )
    write_qa_stats = traits.Bool(desc="Save QA stats.", argstr="--write_qa_stats")
    exclude = InputMultiObject(
        traits.Str,
        desc="List of label IDs to exclude in any output stats files.",
        argstr="--exclude %s",
    )
    keep_ac = traits.Bool(
        desc="Explicitly keep anterior commissure in the volume/qa files.",
        argstr="--keep_ac",
    )
    vox_count_volumes = traits.Bool(
        desc="Use discrete voxel count for label volumes.", argstr="--vox-count-volumes"
    )
    model = File(
        desc="Alternative model weights to load.", exists=True, argstr="--model %s"
    )
    ctab = File(
        desc="Alternative color lookup table to embed in segmentation.",
        exists=True,
        argstr="--ctab %s",
    )
    population_stats = File(
        desc="Alternative population volume stats for QA output.",
        argstr="--population-stats %s",
    )
    debug = traits.Bool(desc="Enable debug logging.", argstr="--debug")
    vmp = traits.Bool(desc="Enable printing of vmpeak at the end.", argstr="--vmp")
    threads = traits.Int(desc="Number of threads to use.", argstr="--threads %d")
    t7 = traits.Bool(desc="Preprocess 7T images.", argstr="--7T")
    percentile = traits.Float(
        desc="Use intensity percentile threshold for normalization.",
        argstr="--percentile %f",
    )
    cuda_device = traits.Int(
        desc="Cuda device for GPU support.", argstr="--cuda-device %d"
    )
    output_base = traits.Str(
        desc="String to use in output file name.", argstr="--output-base %s"
    )
    nchannels = traits.Int(desc="Number of channels", argstr="--nchannels %d")


class MRISclimbicSegOutputSpec(TraitedSpec):
    out_file = File(desc="Segmentation output.")
    out_stats = File(desc="Segmentation stats output.")


class MRISclimbicSeg(CommandLine):
    """Run ``mri_sclimbic_seg`` unless outputs already exist."""

    _cmd = "mri_sclimbic_seg"
    input_spec = MRISclimbicSegInputSpec
    output_spec = MRISclimbicSegOutputSpec

    def _run_interface(self, runtime):
        outputs = self._list_outputs()
        expected = [outputs["out_file"], outputs["out_stats"]]

        if all(os.path.exists(f) for f in expected):
            return runtime

        return super()._run_interface(runtime)

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["out_file"] = os.path.abspath(self.inputs.out_file)
        outputs["out_stats"] = os.path.abspath(self.inputs.out_file).replace(
            ".nii.gz", ".stats"
        )
        return outputs


class SegmentHA_T1InputSpec(FSTraitedSpec):
    subject_id = traits.Str(
        desc="FreeSurfer subject ID", mandatory=True, position=1, argstr="%s"
    )
    subjects_dir = Directory(
        exists=True,
        mandatory=True,
        desc="Path to FreeSurfer subjects directory",
        argstr="%s",
    )


class SegmentHA_T1OutputSpec(TraitedSpec):
    lh_hippoAmygLabels = File(
        exists=True, desc="Left hemisphere hippocampus and amygdala labels"
    )
    rh_hippoAmygLabels = File(
        exists=True, desc="Right hemisphere hippocampus and amygdala labels"
    )
    lh_hippoSfVolumes = File(
        exists=True, desc="Left hemisphere hippocampal subfield volumes"
    )
    lh_amygNucVolumes = File(
        exists=True, desc="Left hemisphere amygdala nuclei volumes"
    )
    rh_hippoAmygLabels = File(
        exists=True, desc="Right hemisphere hippocampus and amygdala labels"
    )
    rh_hippoSfVolumes = File(
        exists=True, desc="Right hemisphere hippocampal subfield volumes"
    )
    rh_amygNucVolumes = File(
        exists=True, desc="Right hemisphere amygdala nuclei volumes"
    )


class SegmentHA_T1(FSCommand):
    """Run ``segmentHA_T1.sh`` unless outputs already exist."""

    _cmd = "segmentHA_T1.sh"
    input_spec = SegmentHA_T1InputSpec
    output_spec = SegmentHA_T1OutputSpec

    def _run_interface(self, runtime):
        fs_path = os.path.join(self.inputs.subjects_dir, self.inputs.subject_id, "mri")
        expected = [
            "lh.hippoAmygLabels-T1.v22.FSvoxelSpace.mgz",
            "rh.hippoAmygLabels-T1.v22.FSvoxelSpace.mgz",
            "lh.hippoSfVolumes-T1.v22.txt",
            "lh.amygNucVolumes-T1.v22.txt",
            "rh.hippoSfVolumes-T1.v22.txt",
            "rh.amygNucVolumes-T1.v22.txt",
        ]

        if all(os.path.exists(os.path.join(fs_path, f)) for f in expected):
            return runtime

        cmd = CommandLine(
            command="segmentHA_T1.sh",
            args=self.inputs.subject_id,
            environ={"SUBJECTS_DIR": self.inputs.subjects_dir},
        )
        runtime = cmd.run()

        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        subj_dir = os.path.abspath(
            self.inputs.subjects_dir + "/" + self.inputs.subject_id + "/mri/"
        )

        outputs["lh_hippoAmygLabels"] = os.path.join(
            subj_dir, "lh.hippoAmygLabels-T1.v22.FSvoxelSpace.mgz"
        )
        outputs["rh_hippoAmygLabels"] = os.path.join(
            subj_dir, "rh.hippoAmygLabels-T1.v22.FSvoxelSpace.mgz"
        )
        outputs["lh_hippoSfVolumes"] = os.path.join(
            subj_dir, "lh.hippoSfVolumes-T1.v22.txt"
        )
        outputs["lh_amygNucVolumes"] = os.path.join(
            subj_dir, "lh.amygNucVolumes-T1.v22.txt"
        )
        outputs["rh_hippoSfVolumes"] = os.path.join(
            subj_dir, "rh.hippoSfVolumes-T1.v22.txt"
        )
        outputs["rh_amygNucVolumes"] = os.path.join(
            subj_dir, "rh.amygNucVolumes-T1.v22.txt"
        )

        return outputs

    def _gen_filename(self, name):
        if name == "subjects_dir":
            return os.path.abspath(self.inputs.subject_id)
        return None


class SegmentThalamicNucleiInputSpec(BaseInterfaceInputSpec):
    subjects_dir = Directory(exists=True, mandatory=True, desc="FreeSurfer subjects directory")
    subject_id = traits.Str(mandatory=True, desc="Subject identifier")


class SegmentThalamicNucleiOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="Thalamic nuclei segmentation in FS voxel space")
    volumes_file = File(exists=True, desc="Thalamic nuclei volume statistics")


class SegmentThalamicNuclei(SimpleInterface):
    """Run ``segmentThalamicNuclei.sh`` unless outputs already exist."""

    input_spec = SegmentThalamicNucleiInputSpec
    output_spec = SegmentThalamicNucleiOutputSpec

    def _run_interface(self, runtime):
        subj_dir = Path(self.inputs.subjects_dir) / self.inputs.subject_id / "mri"
        out_file = subj_dir / "ThalamicNuclei.v13.T1.FSvoxelSpace.mgz"
        volumes_file = subj_dir / "ThalamicNuclei.v13.T1.volumes.txt"

        if not (out_file.exists() and volumes_file.exists()):
            cmd = [
                "segmentThalamicNuclei.sh",
                self.inputs.subject_id,
                str(self.inputs.subjects_dir),
            ]
            self._run_command(cmd)
        else:
            runtime.returncode = 0

        return runtime

    def _run_command(self, cmd):
        import subprocess
        subprocess.run(cmd, check=True, capture_output=True, text=True)

    def _list_outputs(self):
        outputs = self._outputs().get()
        subj_dir = os.path.join(self.inputs.subjects_dir, self.inputs.subject_id, "mri")

        outputs["out_file"] = os.path.join(subj_dir, "ThalamicNuclei.v13.T1.FSvoxelSpace.mgz")
        outputs["volumes_file"] = os.path.join(subj_dir, "ThalamicNuclei.v13.T1.volumes.txt")
        return outputs


class SegmentWMInputSpec(BaseInterfaceInputSpec):
    subjects_dir = Directory(exists=True, mandatory=True, desc='FreeSurfer subjects directory')
    subject_id = traits.Str(mandatory=True, desc='Subject identifier')


class SegmentWMOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='White-matter parcellation')


class SegmentWM(SimpleInterface):
    """Run ``mri_wmparc`` if ``wmparc.mgz`` is missing."""

    input_spec = SegmentWMInputSpec
    output_spec = SegmentWMOutputSpec

    def _run_interface(self, runtime):
        subj_dir = Path(self.inputs.subjects_dir) / self.inputs.subject_id / 'mri'
        out_file = subj_dir / 'wmparc.mgz'

        if not out_file.exists():
            cmd = [
                'mri_wmparc',
                '--s',
                self.inputs.subject_id,
                '--sd',
                str(self.inputs.subjects_dir),
            ]
            self._results['stdout'], self._results['stderr'] = self._run_command(cmd)
        else:
            runtime.returncode = 0

        self._results['out_file'] = str(out_file)
        return runtime

    def _run_command(self, cmd):
        import subprocess

        proc = subprocess.run(cmd, capture_output=True, text=True)
        return proc.stdout, proc.stderr


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