import nibabel as nb
import numpy as np
import pandas as pd
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    traits,
)
from nipype.utils.filemanip import fname_presuffix


class ExtractTACsInputSpec(BaseInterfaceInputSpec):
    pet_file = File(
        exists=True,
        mandatory=True,
        desc='3D or 4D PET image',
    )
    segmentation = File(exists=True, mandatory=True, desc='Segmentation in PET space')
    dseg_tsv = File(exists=True, mandatory=True, desc='Segmentation TSV file')
    metadata = traits.Dict(desc='PET metadata')
    out_file = File(desc='Output TSV file')


class ExtractTACsOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='Extracted TACs TSV file')


class ExtractTACs(SimpleInterface):
    """Extract time-activity curves from a segmentation.

    Both 3D (single-frame) and 4D PET images are accepted.
    """

    input_spec = ExtractTACsInputSpec
    output_spec = ExtractTACsOutputSpec

    def _run_interface(self, runtime):
        pet_img = nb.load(self.inputs.pet_file)
        seg_img = nb.load(self.inputs.segmentation)

        pet_data = np.asanyarray(pet_img.dataobj)
        if pet_data.ndim == 3:
            pet_data = pet_data[..., np.newaxis]
        if pet_data.ndim != 4:
            raise RuntimeError(
                f"Expected a 3D or 4D PET image, got {pet_data.ndim} dimensions"
            )

        seg_data = np.asanyarray(seg_img.dataobj).astype(int)

        df_labels = pd.read_csv(self.inputs.dseg_tsv, sep='\t')
        labels = df_labels.iloc[:, 0].astype(int).to_list()
        names = df_labels['name'].to_list()

        nframes = pet_data.shape[3]
        meta = self.inputs.metadata or {}
        frame_starts = meta.get('FrameTimesStart')
        frame_durs = meta.get('FrameDuration')

        if not isinstance(frame_starts, (list, tuple)) and frame_starts is not None:
            frame_starts = [frame_starts]
        if not isinstance(frame_durs, (list, tuple)) and frame_durs is not None:
            frame_durs = [frame_durs]

        if frame_starts is None:
            frame_starts = [0] if nframes == 1 else list(range(nframes))
        if frame_durs is None:
            frame_durs = [1] * nframes

        frame_starts = list(frame_starts)[:nframes]
        frame_durs = list(frame_durs)[:nframes]
        frame_ends = [s + d for s, d in zip(frame_starts, frame_durs, strict=False)]

        out_df = pd.DataFrame(
            {
                'FrameTimeStart': frame_starts,
                'FrameTimesEnd': frame_ends,
            }
        )

        flat_pet = pet_data.reshape(-1, nframes)
        flat_seg = seg_data.reshape(-1)
        for label, name in zip(labels, names, strict=False):
            mask = flat_seg == label
            if not np.any(mask):
                out_df[name] = [np.nan] * nframes
            else:
                out_df[name] = flat_pet[mask].mean(axis=0)

        out_file = self.inputs.out_file
        if not out_file:
            out_file = (
                fname_presuffix(
                    self.inputs.pet_file,
                    suffix='_tacs',
                    newpath=runtime.cwd,
                    use_ext=False,
                )
                + '.tsv'
            )

        out_df.to_csv(out_file, sep='\t', index=False, float_format='%.8g')
        self._results['out_file'] = out_file
        return runtime

