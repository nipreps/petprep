import nibabel as nb
import numpy as np
from nipype.interfaces.base import (
    BaseInterface,
    BaseInterfaceInputSpec,
    TraitedSpec,
    File,
)
import os


class Binarise4DSegmentationInputSpec(BaseInterfaceInputSpec):
    dseg_file = File(exists=True, mandatory=True, desc="Input segmentation file (_dseg.nii.gz)")
    out_file = File("binarised_4d.nii.gz", usedefault=True, desc="Output 4D binary segmentation")


class Binarise4DSegmentationOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="Output 4D binary segmentation file")


class Binarise4DSegmentation(BaseInterface):
    input_spec = Binarise4DSegmentationInputSpec
    output_spec = Binarise4DSegmentationOutputSpec

    def _run_interface(self, runtime):
        dseg_img = nb.load(self.inputs.dseg_file)
        dseg_data = dseg_img.get_fdata().astype(np.int32)

        region_labels = np.unique(dseg_data)
        region_labels = region_labels[region_labels != 0]  # exclude background

        binarised_4d = np.zeros(dseg_data.shape + (len(region_labels),), dtype=np.uint8)

        for idx, label in enumerate(region_labels):
            binarised_4d[..., idx] = (dseg_data == label).astype(np.uint8)

        new_img = nb.Nifti1Image(binarised_4d, affine=dseg_img.affine, header=dseg_img.header)
        nb.save(new_img, os.path.abspath(self.inputs.out_file))

        return runtime

    def _list_outputs(self):
        return {"out_file": os.path.abspath(self.inputs.out_file)}
    

class StackProbSegMapsInputSpec(BaseInterfaceInputSpec):
    probseg_files = InputMultiPath(
        File(exists=True),
        mandatory=True,
        desc="List of tissue probability segmentation maps (_probseg.nii.gz).",
    )
    out_file = File(
        "stacked_probseg_maps.nii.gz",
        usedefault=True,
        desc="Output 4D stacked probability segmentation maps.",
    )


class StackProbSegMapsOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="Output 4D stacked probability segmentation maps.")


class StackProbSegMaps(BaseInterface):
    input_spec = StackProbSegMapsInputSpec
    output_spec = StackProbSegMapsOutputSpec

    def _run_interface(self, runtime):
        images = [nb.load(img_file) for img_file in self.inputs.probseg_files]

        shapes = [img.shape for img in images]
        affines = [img.affine for img in images]

        if len(set(shapes)) > 1 or len(set([tuple(a.flatten()) for a in affines])) > 1:
            raise ValueError("All probability maps must have the same shape and affine.")

        stacked_data = np.stack([img.get_fdata() for img in images], axis=-1)

        stacked_img = nb.Nifti1Image(
            stacked_data, affine=images[0].affine, header=images[0].header
        )
        nb.save(stacked_img, os.path.abspath(self.inputs.out_file))

        return runtime

    def _list_outputs(self):
        return {"out_file": os.path.abspath(self.inputs.out_file)}

