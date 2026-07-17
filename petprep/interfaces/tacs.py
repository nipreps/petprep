import json
import os

import nibabel as nb
import numpy as np
import pandas as pd
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    isdefined,
    traits,
)
from nipype.utils.filemanip import fname_presuffix


class _ExtractTACsInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='PET file in anatomical space')
    segmentation = File(exists=True, mandatory=True, desc='Segmentation in anatomical space')
    dseg_tsv = File(exists=True, mandatory=True, desc='Lookup table for segmentation')
    metadata = File(exists=True, mandatory=True, desc='PET JSON metadata file')


class _ExtractTACsOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='Regional time activity curves')


class ExtractTACs(SimpleInterface):
    """Extract time activity curves from a segmentation."""

    input_spec = _ExtractTACsInputSpec
    output_spec = _ExtractTACsOutputSpec

    def _run_interface(self, runtime):
        pet_img = nb.load(self.inputs.in_file)
        if pet_img.ndim == 3:
            pet_img = nb.Nifti1Image(
                pet_img.get_fdata()[..., np.newaxis], pet_img.affine, pet_img.header
            )

        seginfo = pd.read_csv(self.inputs.dseg_tsv, sep='\t', dtype={0: str, 1: str})
        label_mapping = dict(zip(seginfo.iloc[:, 0], seginfo.iloc[:, 1], strict=False))

        with open(self.inputs.metadata) as f:
            metadata = json.load(f)

        frame_times = metadata.get('FrameTimesStart', [])
        frame_durations = metadata.get('FrameDuration', [])

        if len(frame_times) != len(frame_durations):
            raise ValueError('FrameTimesStart and FrameDuration must have equal length')

        segmentation_data = np.rint(nb.load(self.inputs.segmentation).get_fdata()).astype(int)
        pet_data = pet_img.get_fdata()

        unique_labels = np.unique(segmentation_data)
        n_tp = pet_data.shape[-1]
        if len(frame_times) != n_tp:
            raise ValueError(
                'Number of PET frames does not match FrameTimesStart/FrameDuration length'
            )

        curves = {}

        for label_num in unique_labels:
            if label_num == 0:
                continue  # Skip background
            label_key = str(label_num)
            label_name = label_mapping.get(label_key, f'label_{label_num}')
            mask = segmentation_data == label_num
            if mask.any():
                region_timeseries = pet_data[mask, :].mean(axis=0)
                curves[label_name] = region_timeseries
            else:
                curves[label_name] = np.full(n_tp, np.nan)

        frame_times_end = np.add(frame_times, frame_durations).tolist()
        df = pd.DataFrame(curves)
        df.insert(0, 'frame_end', frame_times_end)
        df.insert(0, 'frame_start', list(frame_times))

        out_file = fname_presuffix(
            self.inputs.in_file,
            suffix='_tacs.tsv',
            newpath=runtime.cwd,
            use_ext=False,
        )
        df.to_csv(out_file, sep='\t', index=False, na_rep='n/a')

        self._results['out_file'] = out_file
        return runtime


class _ExtractRefTACInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='PET file in anatomical space')
    mask_file = File(exists=True, mandatory=True, desc='Reference mask in anatomical space')
    ref_mask_name = traits.Str(mandatory=True, desc='Name of reference region')
    metadata = File(exists=True, mandatory=True, desc='PET JSON metadata file')


class _ExtractRefTACOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='Reference region time activity curve')


class ExtractRefTAC(SimpleInterface):
    """Extract a time activity curve from a reference mask."""

    input_spec = _ExtractRefTACInputSpec
    output_spec = _ExtractRefTACOutputSpec

    def _run_interface(self, runtime):
        pet_img = nb.load(self.inputs.in_file)
        pet_data = pet_img.get_fdata()
        if pet_img.ndim == 3:
            pet_data = pet_data[..., np.newaxis]

        mask = np.rint(nb.load(self.inputs.mask_file).get_fdata()).astype(np.int16) > 0

        with open(self.inputs.metadata) as f:
            metadata = json.load(f)

        frame_times = metadata.get('FrameTimesStart', [])
        frame_durations = metadata.get('FrameDuration', [])

        if len(frame_times) != len(frame_durations):
            raise ValueError('FrameTimesStart and FrameDuration must have equal length')

        n_tp = pet_data.shape[-1]
        if len(frame_times) != n_tp:
            raise ValueError(
                'Number of PET frames does not match FrameTimesStart/FrameDuration length'
            )

        timeseries = pet_data[mask, :].mean(axis=0)
        frame_times_end = np.add(frame_times, frame_durations).tolist()
        df = pd.DataFrame({self.inputs.ref_mask_name: timeseries})
        df.insert(0, 'frame_end', frame_times_end)
        df.insert(0, 'frame_start', list(frame_times))

        out_file = fname_presuffix(
            self.inputs.in_file,
            suffix='_tacs.tsv',
            newpath=runtime.cwd,
            use_ext=False,
        )
        df.to_csv(out_file, sep='\t', index=False, na_rep='n/a')

        self._results['out_file'] = out_file
        return runtime


class _ReferenceTACPlotInputSpec(BaseInterfaceInputSpec):
    tacs_file = File(exists=True, mandatory=True, desc='Reference-region TAC TSV file')
    confounds_file = File(
        exists=True,
        desc='Optional confounds TSV file containing a global_signal column',
    )


class _ReferenceTACPlotOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='Reference-region TAC plot')


class ReferenceTACPlot(SimpleInterface):
    """Plot a reference-region time-activity curve and optional whole-brain signal."""

    input_spec = _ReferenceTACPlotInputSpec
    output_spec = _ReferenceTACPlotOutputSpec

    def _run_interface(self, runtime):
        import matplotlib as mpl

        mpl.use('Agg', force=True)
        from matplotlib import pyplot as plt

        tacs = pd.read_csv(self.inputs.tacs_file, sep='\t', na_values='n/a')
        timing_columns = {'frame_start', 'frame_end'}
        activity_columns = [column for column in tacs.columns if column not in timing_columns]
        if len(activity_columns) != 1:
            raise ValueError(
                'Reference TAC file must contain frame_start, frame_end, and one activity column'
            )

        missing_timing = timing_columns.difference(tacs.columns)
        if missing_timing:
            raise ValueError(
                f'Reference TAC file is missing timing column(s): {sorted(missing_timing)}'
            )

        frame_midpoints = (tacs['frame_start'] + tacs['frame_end']) / 2.0
        frame_midpoints_min = frame_midpoints / 60.0
        reference_name = activity_columns[0]

        fig, ax = plt.subplots(figsize=(10, 3.2))
        ax.plot(
            frame_midpoints_min,
            tacs[reference_name],
            color='#D55E00',
            marker='o',
            markersize=4,
            linewidth=1.8,
            label=reference_name.replace('_', ' '),
        )

        if isdefined(self.inputs.confounds_file):
            confounds = pd.read_csv(self.inputs.confounds_file, sep='\t', na_values='n/a')
            if 'global_signal' in confounds.columns and len(confounds) == len(tacs):
                ax.plot(
                    frame_midpoints_min,
                    confounds['global_signal'],
                    color='#0072B2',
                    linewidth=1.5,
                    label='whole brain',
                )

        ax.set_xlabel('Time (min)')
        ax.set_ylabel('Mean PET signal')
        ax.grid(axis='y', color='0.9', linewidth=0.8)
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(frameon=False)
        fig.tight_layout()

        out_file = os.path.abspath('reference_tac.svg')
        fig.savefig(out_file, format='svg', bbox_inches='tight')
        plt.close(fig)
        self._results['out_file'] = out_file
        return runtime


__all__ = ('ExtractTACs', 'ExtractRefTAC', 'ReferenceTACPlot')
