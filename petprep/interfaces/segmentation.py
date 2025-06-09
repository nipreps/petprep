from __future__ import annotations

from pathlib import Path

from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    Directory,
    File,
    SimpleInterface,
    TraitedSpec,
    traits,
)


class SegmentBSInputSpec(BaseInterfaceInputSpec):
    subjects_dir = Directory(exists=True, mandatory=True, desc='FreeSurfer subjects directory')
    subject_id = traits.Str(mandatory=True, desc='Subject identifier')


class SegmentBSOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='Brainstem segmentation in anatomical space')
    out_fsvox_file = File(exists=True, desc='Brainstem segmentation in FS voxel space')
    volumes_file = File(exists=True, desc='Brainstem volumes table')


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