import nibabel as nb
import numpy as np
import pandas as pd
from nipype.interfaces.base import (
    BaseInterface,
    BaseInterfaceInputSpec,
    TraitedSpec,
    File,
    traits,
    InputMultiPath,
)
import os


class Binarise4DSegmentationInputSpec(BaseInterfaceInputSpec):
    dseg_file = File(exists=True, mandatory=True, desc="Input segmentation file (_dseg.nii.gz)")
    out_file = File("binarised_4d.nii.gz", usedefault=True, desc="Output 4D binary segmentation")


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


class StackTissueProbabilityMapsInputSpec(BaseInterfaceInputSpec):
    t1w_tpms = traits.List(File(exists=True), mandatory=True, desc="List of T1w tissue probability maps")
    out_file = File("stacked_probseg.nii.gz", usedefault=True, desc="Output stacked 4D probability map")


class StackTissueProbabilityMapsOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="Output stacked 4D probability map")


class StackTissueProbabilityMaps(BaseInterface):
    input_spec = StackTissueProbabilityMapsInputSpec
    output_spec = StackTissueProbabilityMapsOutputSpec

    def _run_interface(self, runtime):
        imgs = [nb.load(f).get_fdata() for f in self.inputs.t1w_tpms]
        affine = nb.load(self.inputs.t1w_tpms[0]).affine

        stacked_img = np.stack(imgs, axis=-1)

        new_img = nb.Nifti1Image(stacked_img, affine=affine)
        nb.save(new_img, os.path.abspath(self.inputs.out_file))

        return runtime

    def _list_outputs(self):
        return {"out_file": os.path.abspath(self.inputs.out_file)}


class CSVtoNiftiInputSpec(BaseInterfaceInputSpec):
    csv_file = File(exists=True, mandatory=True, desc="Input CSV file with region means")
    reference_nifti = File(exists=True, mandatory=True, desc="Reference NIfTI file for spatial information")
    label_list = traits.List(traits.Int, mandatory=True, desc="List of labels corresponding to regions")
    out_file = File("output_from_csv.nii.gz", usedefault=True, desc="Output NIfTI file")


class CSVtoNiftiOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="Output NIfTI image file")


class CSVtoNifti(BaseInterface):
    input_spec = CSVtoNiftiInputSpec
    output_spec = CSVtoNiftiOutputSpec

    def _run_interface(self, runtime):
        import pandas as pd

        csv_data = pd.read_csv(self.inputs.csv_file, sep='\t')
        reference_img = nb.load(self.inputs.reference_nifti)
        reference_data = reference_img.get_fdata().astype(np.int32)

        output_data = np.zeros(reference_data.shape, dtype=np.float32)

        for idx, label in enumerate(self.inputs.label_list):
            mean_value = csv_data.iloc[idx]['MEAN']
            output_data[reference_data == label] = mean_value

        output_img = nb.Nifti1Image(output_data, affine=reference_img.affine, header=reference_img.header)
        nb.save(output_img, os.path.abspath(self.inputs.out_file))

        return runtime

    def _list_outputs(self):
        return {"out_file": os.path.abspath(self.inputs.out_file)}
