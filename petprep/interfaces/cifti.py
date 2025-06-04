from pathlib import Path
import json

from niworkflows.interfaces.cifti import (
    _GenerateCiftiOutputSpec,
    _prepare_cifti,
    _create_cifti_image,
)
from nipype.interfaces.base import BaseInterfaceInputSpec, File, SimpleInterface, TraitedSpec, traits


class _GeneratePetCiftiInputSpec(BaseInterfaceInputSpec):
    pet_file = File(mandatory=True, exists=True, desc="input PET file")
    volume_target = traits.Enum(
        "MNI152NLin6Asym",
        usedefault=True,
        desc="CIFTI volumetric output space",
    )
    surface_target = traits.Enum(
        "fsLR",
        usedefault=True,
        desc="CIFTI surface target space",
    )
    grayordinates = traits.Enum("91k", "170k", usedefault=True, desc="Final CIFTI grayordinates")
    TR = traits.Float(mandatory=True, desc="Repetition time")
    surface_pets = traits.List(
        File(exists=True),
        mandatory=True,
        desc="list of surface PET GIFTI files (length 2 with order [L,R])",
    )


class GeneratePetCifti(SimpleInterface):
    """Generate a HCP-style CIFTI image from a PET file in target spaces."""

    input_spec = _GeneratePetCiftiInputSpec
    output_spec = _GenerateCiftiOutputSpec

    def _run_interface(self, runtime):
        surface_labels, volume_labels, metadata = _prepare_cifti(self.inputs.grayordinates)
        self._results["out_file"] = _create_cifti_image(
            self.inputs.pet_file,
            volume_labels,
            self.inputs.surface_pets,
            surface_labels,
            self.inputs.TR,
            metadata,
        )
        metadata_file = Path("pet.dtseries.json").absolute()
        metadata_file.write_text(json.dumps(metadata, indent=2))
        self._results["out_metadata"] = str(metadata_file)
        return runtime


__all__ = ("GeneratePetCifti",)