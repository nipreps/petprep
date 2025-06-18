import nibabel as nb
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    traits,
)
from nipype.utils.filemanip import fname_presuffix


class GTMPVCInputSpec(BaseInterfaceInputSpec):
    pet_file = File(exists=True, mandatory=True, desc='Input PET image')
    petref = File(exists=True, mandatory=True, desc='Input PET reference image')
    mask_file = File(exists=True, desc='Mask or segmentation in same space as pet_file')
    method = traits.Str('none', usedefault=True, desc='PVC method')
    fwhm = traits.Any(desc='Point-spread function FWHM')


class GTMPVCOutputSpec(TraitedSpec):
    out_file = File(desc='PVC corrected image')


class GTMPVC(SimpleInterface):
    """Placeholder interface for partial volume correction."""

    input_spec = PVCInputSpec
    output_spec = PVCOutputSpec

    def _run_interface(self, runtime):
        out_file = fname_presuffix(self.inputs.in_file, suffix='_pvc', newpath=runtime.cwd)
        nb.load(self.inputs.in_file).to_filename(out_file)
        self._results['out_file'] = out_file
        return runtime
    

class PETPVCInputSpec(BaseInterfaceInputSpec):
    pet_file = File(exists=True, mandatory=True, desc='Input PET image')
    petref = File(exists=True, mandatory=True, desc='Input PET reference image')
    mask_file = File(exists=True, desc='Mask or segmentation in same space as pet_file')
    method = traits.Str('none', usedefault=True, desc='PVC method')
    fwhm = traits.Any(desc='Point-spread function FWHM')


class PETPVCOutputSpec(TraitedSpec):
    out_file = File(desc='PVC corrected image')


class PETPVC(SimpleInterface):
    """Placeholder interface for partial volume correction."""

    input_spec = PVCInputSpec
    output_spec = PVCOutputSpec

    def _run_interface(self, runtime):
        out_file = fname_presuffix(self.inputs.in_file, suffix='_pvc', newpath=runtime.cwd)
        nb.load(self.inputs.in_file).to_filename(out_file)
        self._results['out_file'] = out_file
        return runtime