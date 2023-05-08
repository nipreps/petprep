import os
from nipype.interfaces.base import (
    BaseInterface,
    BaseInterfaceInputSpec,
    TraitedSpec,
    CommandLine,
    File,
    Directory,
    traits,
)

class SegmentBSInputSpec(BaseInterfaceInputSpec):
    subjects_dir = Directory(
        desc="FreeSurfer subjects directory", exists=True, mandatory=True
    )
    subject_id = traits.Str(
        desc="Subject ID", mandatory=True
    )

class SegmentBSOutputSpec(TraitedSpec):
    fs_voxel_space_file = File(
        desc="Output file brainstemSsLabels.v13.FSvoxelSpace.mgz"
    )
    mgz_file = File(
        desc="Output file brainstemSsLabels.v13.mgz"
    )
    volumes_txt = File(
        desc="Output file brainstemSsVolumes.v13.txt"
    )

class SegmentBS(BaseInterface):
    input_spec = SegmentBSInputSpec
    output_spec = SegmentBSOutputSpec

    def _run_interface(self, runtime):
        subjects_dir = self.inputs.subjects_dir
        subject_id = self.inputs.subject_id

        cmd = CommandLine(
            command="segmentBS.sh",
            args=subject_id,
            environ={"SUBJECTS_DIR": subjects_dir},
        )
        runtime = cmd.run()

        return runtime

    def _list_outputs(self):
        fs_path = os.path.join(
            self.inputs.subjects_dir, self.inputs.subject_id, "mri"
        )
        outputs = self._outputs().get()
        outputs["fs_voxel_space_file"] = os.path.join(
            fs_path, "brainstemSsLabels.v13.FSvoxelSpace.mgz"
        )
        outputs["mgz_file"] = os.path.join(
            fs_path, "brainstemSsLabels.v13.mgz"
        )
        outputs["volumes_txt"] = os.path.join(
            fs_path, "brainstemSsVolumes.v13.txt"
        )
        return outputs