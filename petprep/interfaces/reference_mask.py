import json
import os

import nibabel as nib
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    isdefined,
    traits,
)


class ExtractRefRegionInputSpec(BaseInterfaceInputSpec):
    seg_file = File(exists=True, mandatory=True, desc='Segmentation NIfTI file')
    gm_probseg = File(exists=True, desc='Gray matter probability map for thresholding')
    config_file = File(exists=True, mandatory=True, desc='Path to the config.json file')
    segmentation_type = traits.Str(mandatory=True, desc="Type of segmentation (e.g. 'gtm', 'wm')")
    region_name = traits.Str(
        mandatory=True, desc="Name of the reference region (e.g. 'cerebellum')"
    )
    override_indices = traits.List(traits.Int, desc='Use these indices instead of configuration')


class ExtractRefRegionOutputSpec(TraitedSpec):
    refmask_file = File(exists=True, desc='Output reference mask NIfTI file')


class ExtractRefRegion(SimpleInterface):
    input_spec = ExtractRefRegionInputSpec
    output_spec = ExtractRefRegionOutputSpec

    def _run_interface(self, runtime):
        seg_img = nib.load(self.inputs.seg_file)
        gm_prob_img = None
        if isdefined(self.inputs.gm_probseg):
            gm_prob_img = nib.load(self.inputs.gm_probseg)

        if isdefined(self.inputs.override_indices):
            cfg = {'refmask_indices': list(self.inputs.override_indices)}
        else:
            # Load the config
            with open(self.inputs.config_file) as f:
                config = json.load(f)

            try:
                cfg = config[self.inputs.segmentation_type][self.inputs.region_name]
            except KeyError as err:
                raise ValueError(
                    f"Configuration not found for segmentation='{self.inputs.segmentation_type}' "
                    f"and region='{self.inputs.region_name}'"
                ) from err

        from petprep.utils.reference_mask import generate_reference_region

        refmask_img = generate_reference_region(
            seg_img=seg_img,
            config=cfg,
            gm_probseg_img=gm_prob_img,
        )

        out_file = os.path.abspath('refmask.nii.gz')
        nib.save(refmask_img, out_file)
        self._results['refmask_file'] = out_file
        return runtime
