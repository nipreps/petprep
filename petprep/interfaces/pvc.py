import nibabel as nb
import numpy as np
import pandas as pd
from nipype.interfaces.base import (
    BaseInterface,
    BaseInterfaceInputSpec,
    TraitedSpec,
    File,
    traits,
)
import os


class Binarise4DSegmentationInputSpec(BaseInterfaceInputSpec):
    dseg_file = File(exists=True, mandatory=True, desc="Input segmentation file (_dseg.nii.gz)")


class Binarise4DSegmentationOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="Output 4D binary segmentation file")
    label_list = traits.List(traits.Int, desc="List of labels corresponding to the segmentation regions")


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

        self._label_list = region_labels.tolist()

        return runtime

    def _list_outputs(self):
        return {
            "out_file": os.path.abspath(self.inputs.out_file),
            "label_list": self._label_list
        }


class StackProbSegMapsInputSpec(BaseInterfaceInputSpec):
    probseg_files = InputMultiPath(
        File(exists=True),
        mandatory=True,
        desc="List of tissue probability segmentation maps (_probseg.nii.gz).",
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


class CSVtoNiftiInputSpec(BaseInterfaceInputSpec):
    csv_file = File(exists=True, mandatory=True, desc="Input CSV file with region means")
    reference_nifti = File(exists=True, mandatory=True, desc="Reference NIfTI file for spatial information")
    label_list = traits.List(traits.Int, mandatory=True, desc="List of labels corresponding to regions")


class CSVtoNiftiOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="Output NIfTI image file")


class CSVtoNifti(BaseInterface):
    input_spec = CSVtoNiftiInputSpec
    output_spec = CSVtoNiftiOutputSpec

    def _run_interface(self, runtime):
        csv_data = pd.read_csv(self.inputs.csv_file, sep="\t")
        reference_img = nb.load(self.inputs.reference_nifti)
        reference_data = reference_img.get_fdata().astype(np.int32)

        output_data = np.zeros(reference_data.shape, dtype=np.float32)

        label_means = dict(zip(csv_data['REGION'], csv_data['MEAN']))

        for label in self.inputs.label_list:
            if label in label_means:
                output_data[reference_data == label] = label_means[label]

        output_img = nb.Nifti1Image(output_data, affine=reference_img.affine, header=reference_img.header)
        nb.save(output_img, os.path.abspath(self.inputs.out_file))

        return runtime

    def _list_outputs(self):
        return {"out_file": os.path.abspath(self.inputs.out_file)}
