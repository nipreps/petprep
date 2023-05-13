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
from nipype.interfaces.freesurfer.base import FSCommand, FSTraitedSpec

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
    
class SegmentHA_T1InputSpec(FSTraitedSpec):
    subject_id = traits.Str(desc='FreeSurfer subject ID', mandatory=True, position=1, argstr='%s')
    subjects_dir = Directory(exists=True, mandatory=True, desc="Path to FreeSurfer subjects directory", argstr='%s')

class SegmentHA_T1OutputSpec(TraitedSpec):
    lh_hippoAmygLabels = File(exists=True, desc='Left hemisphere hippocampus and amygdala labels')
    rh_hippoAmygLabels = File(exists=True, desc='Right hemisphere hippocampus and amygdala labels')
    lh_hippoSfVolumes = File(exists=True, desc='Left hemisphere hippocampal subfield volumes')
    lh_amygNucVolumes = File(exists=True, desc='Left hemisphere amygdala nuclei volumes')
    rh_hippoAmygLabels = File(exists=True, desc='Right hemisphere hippocampus and amygdala labels')
    rh_hippoSfVolumes = File(exists=True, desc='Right hemisphere hippocampal subfield volumes')
    rh_amygNucVolumes = File(exists=True, desc='Right hemisphere amygdala nuclei volumes')

class SegmentHA_T1(FSCommand):
    _cmd = 'segmentHA_T1.sh'
    input_spec = SegmentHA_T1InputSpec
    output_spec = SegmentHA_T1OutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        subj_dir = os.path.abspath(self.inputs.subject_id + '/mri/')

        outputs['lh_hippoAmygLabels'] = os.path.join(subj_dir, 'lh.hippoAmygLabels-T1.v22.FSvoxelSpace.mgz')
        outputs['rh_hippoAmygLabels'] = os.path.join(subj_dir, 'rh.hippoAmygLabels-T1.v22.FSvoxelSpace.mgz')
        outputs['lh_hippoSfVolumes'] = os.path.join(subj_dir, 'lh.hippoSfVolumes-T1.v22.txt')
        outputs['lh_amygNucVolumes'] = os.path.join(subj_dir, 'lh.amygNucVolumes-T1.v22.txt')
        outputs['rh_hippoAmygLabels'] = os.path.join(subj_dir, 'rh.hippoAmygLabels-T1.v22.FSvoxelSpace.mgz')
        outputs['rh_hippoSfVolumes'] = os.path.join(subj_dir, 'rh.hippoSfVolumes-T1.v22.txt')
        outputs['rh_amygNucVolumes'] = os.path.join(subj_dir, 'rh.amygNucVolumes-T1.v22.txt')

        return outputs

    def _gen_filename(self, name):
        if name == 'subjects_dir':
            return os.path.abspath(self.inputs.subject_id)
        return None
    
class SegmentThalamicNucleiInputSpec(FSTraitedSpec):
    subject_id = traits.Str(mandatory=True, desc="Subject ID")
    subjects_dir = Directory(exists=True, mandatory=True, desc="Path to FreeSurfer subjects directory")
    additional_vol = traits.Str(desc="Additional volume to be used")
    analysis_id = traits.Str(desc="Unique analysis ID")
    bbregister_mode = traits.Str(desc="Registration mode for bbregister")

class SegmentThalamicNucleiOutputSpec(TraitedSpec):
    thalamicnuc = File(exists=True, desc="Thalamic nuclei segmentation")
    thalamicnuc_volumes = File(exists=True, desc="Volume measurements of thalamic nuclei")

class SegmentThalamicNuclei(FSCommand):
    _cmd = 'segmentThalamicNuclei.sh'
    input_spec = SegmentThalamicNucleiInputSpec
    output_spec = SegmentThalamicNucleiOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        subj_dir = os.path.abspath(self.inputs.subject_id + '/mri/')

        outputs['thalamicnuc'] = os.path.join(subj_dir, 'ThalamicNuclei.v13.T1.FSvoxelSpace.mgz')
        outputs['thalamicnuc_volumes'] = os.path.join(subj_dir, 'ThalamicNuclei.v13.T1.volumes.txt')
        return outputs