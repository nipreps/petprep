from __future__ import annotations

from pathlib import Path
import os

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
)

from nipype.interfaces.freesurfer.base import FSCommand, FSTraitedSpec

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
    subjects_dir = Directory(
        desc="FreeSurfer subjects directory (bids_dir/derivatives/freesurfer)",
        exists=True,
        mandatory=True,
    )
    subject_id = traits.Str(desc="Subject ID (i.e. sub-XX)", mandatory=True)


class SegmentThalamicNucleiOutputSpec(TraitedSpec):
    thalamic_labels_voxel = File(desc="ThalamicNuclei.v13.T1.FSvoxelSpace.mgz")
    thalamic_volumes_txt = File(desc="Output file ThalamicNuclei.v13.T1.volumes.txt")


class SegmentThalamicNuclei(BaseInterface):
    """Run ``segmentThalamicNuclei.sh`` unless outputs already exist."""

    input_spec = SegmentThalamicNucleiInputSpec
    output_spec = SegmentThalamicNucleiOutputSpec

    def _run_interface(self, runtime):
        subjects_dir = self.inputs.subjects_dir
        subject_id = self.inputs.subject_id

        fs_path = os.path.join(subjects_dir, subject_id, "mri")
        expected = [
            "ThalamicNuclei.v13.T1.FSvoxelSpace.mgz",
            "ThalamicNuclei.v13.T1.volumes.txt",
        ]

        if all(os.path.exists(os.path.join(fs_path, f)) for f in expected):
            return runtime

        cmd = CommandLine(
            command="segmentThalamicNuclei.sh",
            args=subject_id,
            environ={"SUBJECTS_DIR": subjects_dir},
        )
        runtime = cmd.run()

        return runtime

    def _list_outputs(self):
        fs_path = os.path.join(self.inputs.subjects_dir, self.inputs.subject_id, "mri")
        outputs = self._outputs().get()
        outputs["thalamic_labels_voxel"] = os.path.join(
            fs_path, "ThalamicNuclei.v13.T1.FSvoxelSpace.mgz"
        )
        outputs["thalamic_volumes_txt"] = os.path.join(
            fs_path, "ThalamicNuclei.v13.T1.volumes.txt"
        )
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
