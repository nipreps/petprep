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
        desc="FreeSurfer subjects directory (bids_dir/derivatives/freesurfer)", exists=True, mandatory=True
    )
    subject_id = traits.Str(
        desc="Subject ID (i.e. sub-XX)", mandatory=True
    )

class SegmentBSOutputSpec(TraitedSpec):
    bs_labels_voxel = File(
        desc="Output file brainstemSsLabels.v13.FSvoxelSpace.mgz"
    )
    bs_labels = File(
        desc="Output file brainstemSsLabels.v13.mgz"
    )
    bs_volumes_txt = File(
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
        outputs["bs_labels_voxel"] = os.path.join(
            fs_path, "brainstemSsLabels.v13.FSvoxelSpace.mgz"
        )
        outputs["bs_labels"] = os.path.join(
            fs_path, "brainstemSsLabels.v13.mgz"
        )
        outputs["bs_volumes_txt"] = os.path.join(
            fs_path, "brainstemSsVolumes.v13.txt"
        )
        return outputs