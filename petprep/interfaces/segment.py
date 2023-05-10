import os
from nipype.interfaces.base import (
    BaseInterface,
    BaseInterfaceInputSpec,
    TraitedSpec,
    CommandLine,
    CommandLineInputSpec,
    File,
    Directory,
    traits,
    InputMultiObject
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
    
    from nipype.interfaces.base import (CommandLine, CommandLineInputSpec,
                                    TraitedSpec, File, traits, isdefined,
                                    InputMultiObject, OutputMultiObject)

class MRISclimbicSegInputSpec(CommandLineInputSpec):
    in_file = File(desc="T1-w image(s) to segment.", exists=True, argstr='--i %s')
    out_file = File(desc="Segmentation output.", genfile=True, argstr='--o %s')
    subjects = InputMultiObject(traits.Str, desc="Process a series of freesurfer recon-all subjects.", argstr='--s %s')
    sd = File(desc="Set the subjects directory.", exists=True, argstr='--sd %s')
    conform = traits.Bool(desc="Resample input to 1mm-iso.", argstr='--conform')
    etiv = traits.Bool(desc="Include eTIV in volume stats.", argstr='--etiv')
    tal = File(desc="Alternative talairach xfm transform for estimating TIV.", exists=True, argstr='--tal %s')
    write_posteriors = traits.Bool(desc="Save the label posteriors.", argstr='--write_posteriors')
    write_volumes = traits.Bool(desc="Save label volume stats.", argstr='--write_volumes')
    write_qa_stats = traits.Bool(desc="Save QA stats.", argstr='--write_qa_stats')
    exclude = InputMultiObject(traits.Str, desc="List of label IDs to exclude in any output stats files.", argstr='--exclude %s')
    keep_ac = traits.Bool(desc="Explicitly keep anterior commissure in the volume/qa files.", argstr='--keep_ac')
    vox_count_volumes = traits.Bool(desc="Use discrete voxel count for label volumes.", argstr='--vox-count-volumes')
    model = File(desc="Alternative model weights to load.", exists=True, argstr='--model %s')
    ctab = File(desc="Alternative color lookup table to embed in segmentation.", exists=True, argstr='--ctab %s')
    population_stats = File(desc="Alternative population volume stats for QA output.", exists=True, argstr='--population-stats %s')
    debug = traits.Bool(desc="Enable debug logging.", argstr='--debug')
    vmp = traits.Bool(desc="Enable printing of vmpeak at the end.", argstr='--vmp')
    threads = traits.Int(desc="Number of threads to use.", argstr='--threads %d')
    t7 = traits.Bool(desc="Preprocess 7T images.", argstr='--7T')
    percentile = traits.Float(desc="Use intensity percentile threshold for normalization.", argstr='--percentile %f')
    cuda_device = traits.Int(desc="Cuda device for GPU support.", argstr='--cuda-device %d')
    output_base = traits.Str(desc="String to use in output file name.", argstr='--output-base %s')
    nchannels = traits.Int(desc="Number of channels", argstr='--nchannels %d')

class MRISclimbicSegOutputSpec(TraitedSpec):
    out_file = File(desc="Segmentation output.", exists=True)
    out_stats = File(desc="Segmentation stats output.", exists=True)

class MRISclimbicSeg(CommandLine):
    _cmd = 'mri_sclimbic_seg'
    input_spec = MRISclimbicSegInputSpec
    output_spec = MRISclimbicSegOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["out_file"] = os.path.abspath(self.inputs.out_file)
        outputs["out_stats"] = os.path.abspath(self.inputs.out_file).replace('.nii.gz', '.stats')
        return outputs