# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
"""Robust atlas registration helpers."""

from __future__ import annotations

from importlib.resources import files
from multiprocessing import cpu_count
from os import path as op
from pathlib import Path

import numpy as np
from nipype.interfaces.ants import AffineInitializer
from nipype.interfaces.ants.registration import RegistrationOutputSpec
from nipype.interfaces.base import (
    BaseInterface,
    BaseInterfaceInputSpec,
    File,
    Str,
    isdefined,
    traits,
)

from .. import config
from .ants import TimedRegistration

LOGGER = config.loggers.workflow


class _AtlasSpatialNormalizationInputSpec(BaseInterfaceInputSpec):
    moving_image = File(exists=True, mandatory=True, desc='T1-weighted moving image')
    moving_mask = File(exists=True, desc='Mask applied to the moving image')
    reference_image = File(exists=True, mandatory=True, desc='Atlas reference template')
    reference_mask = File(exists=True, desc='Mask applied to the reference image')
    settings = traits.List(File(exists=True), desc='Explicit ANTs parameter files to try (JSON)')
    flavor = traits.Enum('precise', 'testing', 'fast', usedefault=True, desc='Parameter preset')
    num_threads = traits.Int(
        cpu_count(), usedefault=True, nohash=True, desc='Number of ITK threads to use'
    )
    explicit_masking = traits.Bool(
        True,
        usedefault=True,
        desc='Apply masks to the images before running ANTs instead of passing masks to ANTs',
    )
    use_histogram_matching = traits.Bool(desc='Override histogram matching in parameter files')
    float = traits.Bool(False, usedefault=True, desc='Use single precision computations')
    initial_moving_transform = File(exists=True, desc='Optional initialization transform')


class _AtlasSpatialNormalizationOutputSpec(RegistrationOutputSpec):
    reference_image = File(exists=True, desc='Reference image used in registration')
    settings_file = File(exists=True, desc='JSON file containing the parameters that succeeded')
    parameter_id = Str(desc='Identifier of the successful parameter set')
    runtime_seconds = traits.Float(desc='Elapsed wall-clock time for the registration stage.')


class AtlasSpatialNormalization(BaseInterface):
    """
    Run atlas registration with robust retry logic inspired by NiWorkflows' ``SpatialNormalization``.

    Parameter files under ``petprep.data.segmentation.config`` are used unless explicitly provided.
    """

    input_spec = _AtlasSpatialNormalizationInputSpec
    output_spec = _AtlasSpatialNormalizationOutputSpec

    def __init__(self, **inputs):
        self.norm = None
        self.retry = 1
        self._reference_image = None
        self._settings_file = None
        self._parameter_id = None
        self.terminal_output = 'file'
        super().__init__(**inputs)

    def _list_outputs(self):
        outputs = self.norm._list_outputs()
        outputs['reference_image'] = self._reference_image
        if self._settings_file:
            outputs['settings_file'] = self._settings_file
        if self._parameter_id:
            outputs['parameter_id'] = self._parameter_id
        return outputs

    def _get_settings(self):
        if isdefined(self.inputs.settings) and self.inputs.settings:
            return [str(Path(p).absolute()) for p in self.inputs.settings]

        cfg_root = files('petprep.data.segmentation.config')
        pattern = f't1w-mni_registration_{self.inputs.flavor}_*.json'
        try:
            settings_iter = cfg_root.glob(pattern)
        except AttributeError:
            # Older importlib_resources.MultiplexedPath lacks glob; fall back to iterdir + fnmatch
            import fnmatch

            settings_iter = (p for p in cfg_root.iterdir() if fnmatch.fnmatch(p.name, pattern))

        settings = sorted(str(path) for path in settings_iter)
        if not settings:
            raise RuntimeError(f'No atlas registration settings found for flavor "{self.inputs.flavor}".')
        return settings

    def _get_ants_args(self):
        args = {
            'moving_image': self.inputs.moving_image,
            'fixed_image': self.inputs.reference_image,
            'initial_moving_transform': self.inputs.initial_moving_transform,
            'num_threads': self.inputs.num_threads,
            'float': self.inputs.float,
            'terminal_output': 'file',
            'write_composite_transform': True,
            'collapse_output_transforms': True,
            'output_warped_image': True,
            'output_inverse_warped_image': True,
        }
        self._reference_image = self.inputs.reference_image

        if isdefined(self.inputs.moving_mask):
            if self.inputs.explicit_masking:
                args['moving_image'] = mask(
                    self.inputs.moving_image,
                    self.inputs.moving_mask,
                    'moving_masked.nii.gz',
                )
                args.pop('moving_image_masks', None)
            else:
                args['moving_image_masks'] = self.inputs.moving_mask

        if isdefined(self.inputs.reference_mask):
            if self.inputs.explicit_masking:
                args['fixed_image'] = mask(
                    self.inputs.reference_image,
                    self.inputs.reference_mask,
                    'fixed_masked.nii.gz',
                )
                args.pop('fixed_image_masks', None)
            else:
                args['fixed_image_masks'] = self.inputs.reference_mask

        return args

    def _run_interface(self, runtime):
        settings_files = self._get_settings()
        ants_args = self._get_ants_args()

        if not isdefined(self.inputs.initial_moving_transform):
            LOGGER.info('Estimating initial transform using AffineInitializer')
            initializer = AffineInitializer(
                fixed_image=ants_args['fixed_image'],
                moving_image=ants_args['moving_image'],
                num_threads=self.inputs.num_threads,
            )
            initializer.resource_monitor = False
            initializer.terminal_output = 'allatonce'
            init_result = initializer.run()
            init_outputs = _write_outputs(init_result.runtime, suffix='.nipype-init')
            if init_outputs:
                LOGGER.info(
                    'Terminal outputs of initialization saved (%s).',
                    ', '.join(init_outputs),
                )
            ants_args['initial_moving_transform'] = init_result.outputs.out_file

        for settings_file in settings_files:
            LOGGER.info('Attempting atlas registration with settings file %s.', settings_file)
            self.norm = TimedRegistration(from_file=settings_file, **ants_args)
            if isdefined(self.inputs.use_histogram_matching):
                LOGGER.info(
                    'Overriding (%sabling) histogram matching for file %s',
                    'en' if self.inputs.use_histogram_matching else 'dis',
                    settings_file,
                )
                self.norm.inputs.use_histogram_matching = self.inputs.use_histogram_matching

            self.norm.resource_monitor = False
            self.norm.terminal_output = self.terminal_output
            self.norm.ignore_exception = True

            with open('command.txt', 'w') as cmdfile:
                cmdfile.write(self.norm.cmdline + '\n')

            result = self.norm.run()
            if result.runtime.returncode != 0:
                LOGGER.warning('Atlas registration retry #%d failed.', self.retry)
                term_out = _write_outputs(result.runtime, suffix=f'.nipype-{self.retry:04d}')
                if term_out:
                    LOGGER.warning('Log of failed retry saved (%s).', ', '.join(term_out))
                self.retry += 1
                continue

            runtime.returncode = 0
            LOGGER.info('Atlas registration succeeded on retry #%d.', self.retry)
            self._settings_file = settings_file
            self._parameter_id = Path(settings_file).stem
            return runtime

        raise RuntimeError(f'Atlas spatial normalization failed after {self.retry - 1} retries.')


def mask(in_file, mask_file, new_name):
    """Apply a binary mask to an image."""
    import os

    import nibabel as nb
    import numpy as np

    try:
        from nilearn.image import resample_to_img
    except Exception as exc:  # pragma: no cover - nilearn import should normally succeed
        resample_to_img = None
        LOGGER.warning('Could not import nilearn.image.resample_to_img (%s); proceeding without resampling.', exc)

    in_img = nb.load(in_file)
    mask_img = nb.load(mask_file)

    # Resample mask to the moving image grid if needed.
    if resample_to_img and (in_img.shape != mask_img.shape or not np.allclose(in_img.affine, mask_img.affine)):
        mask_img = resample_to_img(mask_img, in_img, interpolation='nearest', copy=True)

    data = in_img.get_fdata()
    mask_data = np.asanyarray(mask_img.dataobj)
    if mask_data.shape != data.shape:
        raise ValueError(
            f'Mask grid mismatch after resampling: image shape {data.shape}, mask shape {mask_data.shape}'
        )
    data[mask_data == 0] = 0
    masked_img = nb.Nifti1Image(data, in_img.affine, in_img.header)
    masked_img.to_filename(new_name)
    return os.path.abspath(new_name)


def _write_outputs(runtime, suffix='.nipype'):
    out_files = []
    for stream_name in ('stdout', 'stderr', 'merged'):
        stream = getattr(runtime, stream_name, '')
        if not stream:
            continue
        out_file = op.join(runtime.cwd, stream_name + suffix)
        with open(out_file, 'w') as outf:
            print(stream, file=outf)
        out_files.append(out_file)
    return out_files


__all__ = ('AtlasSpatialNormalization',)
