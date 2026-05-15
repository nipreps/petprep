# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Registration helper interfaces."""

from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    isdefined,
    traits,
)


class PETCoregistrationFallbackInputSpec(BaseInterfaceInputSpec):
    ref_pet_brain = File(exists=True, mandatory=True)
    anat_preproc = File(exists=True, mandatory=True)
    anat_mask = File(exists=True, mandatory=True)

    cropped_transform = File(exists=True)
    cropped_inv_transform = File(exists=True)
    cropped_winner = traits.Either(None, traits.Str(), usedefault=True)
    cropped_score = traits.Either(None, traits.Float(), usedefault=True)

    cropped_ants_transform = File(exists=True)
    cropped_fs_transform = File(exists=True)
    cropped_ants_inv_transform = File(exists=True)
    cropped_fs_inv_transform = File(exists=True)
    cropped_ants_score = traits.Either(None, traits.Float(), usedefault=True)
    cropped_fs_score = traits.Either(None, traits.Float(), usedefault=True)

    fallback_threshold = traits.Float(-0.05, usedefault=True)
    pet2anat_dof = traits.Enum(6, 9, 12, mandatory=True)
    pet2anat_method = traits.Enum('mri_coreg', 'robust', 'ants', 'auto', mandatory=True)
    mem_gb = traits.Float(mandatory=True)
    omp_nthreads = traits.Int(mandatory=True)
    sloppy = traits.Bool(False, usedefault=True)


class PETCoregistrationFallbackOutputSpec(TraitedSpec):
    best_transform = File(exists=True)
    best_inv_transform = File(exists=True)
    best_winner = traits.Either(None, traits.Str())
    best_score = traits.Either(None, traits.Float())
    fallback = traits.Bool()
    anat_reference = traits.Enum('cropped', 'uncropped')
    registration_winner = traits.Either(None, traits.Str())
    registration_score = traits.Either(None, traits.Float())


class PETCoregistrationFallback(SimpleInterface):
    """Select cropped registration or lazily run an uncropped fallback."""

    input_spec = PETCoregistrationFallbackInputSpec
    output_spec = PETCoregistrationFallbackOutputSpec

    def _run_interface(self, runtime):
        if self.inputs.pet2anat_method == 'auto':
            cropped = self._select_cropped_auto()
            should_fallback = not any(
                score is not None and score <= self.inputs.fallback_threshold
                for score in (self.inputs.cropped_ants_score, self.inputs.cropped_fs_score)
            )
        else:
            cropped = self._select_cropped_manual()
            should_fallback = not (
                self.inputs.cropped_score is not None
                and self.inputs.cropped_score <= self.inputs.fallback_threshold
            )

        if not should_fallback:
            self._set_results(*cropped, fallback=False, anat_reference='cropped')
            return runtime

        self._require_defined('ref_pet_brain', 'anat_preproc', 'anat_mask')
        uncropped = self._run_uncropped_fallback(runtime.cwd)
        if uncropped[3] is not None and (cropped[3] is None or uncropped[3] < cropped[3]):
            self._set_results(*uncropped, fallback=True, anat_reference='uncropped')
        else:
            self._set_results(*cropped, fallback=False, anat_reference='cropped')

        return runtime

    def _select_cropped_auto(self):
        from petprep.workflows.pet.registration import _select_best_transform

        self._require_defined(
            'cropped_ants_transform',
            'cropped_fs_transform',
            'cropped_ants_inv_transform',
            'cropped_fs_inv_transform',
            'cropped_ants_score',
            'cropped_fs_score',
        )
        return _select_best_transform(
            self.inputs.cropped_ants_transform,
            self.inputs.cropped_fs_transform,
            self.inputs.cropped_ants_inv_transform,
            self.inputs.cropped_fs_inv_transform,
            self.inputs.cropped_ants_score,
            self.inputs.cropped_fs_score,
        )

    def _select_cropped_manual(self):
        self._require_defined('cropped_transform', 'cropped_inv_transform', 'cropped_score')
        return (
            self.inputs.cropped_transform,
            self.inputs.cropped_inv_transform,
            self.inputs.cropped_winner,
            self.inputs.cropped_score,
        )

    def _run_uncropped_fallback(self, cwd):
        import os
        from pathlib import Path

        from nipype.utils.filemanip import loadpkl

        from petprep.workflows.pet.registration import init_pet_reg_wf

        fallback_wf = init_pet_reg_wf(
            pet2anat_dof=self.inputs.pet2anat_dof,
            mem_gb=self.inputs.mem_gb,
            omp_nthreads=self.inputs.omp_nthreads,
            pet2anat_method=self.inputs.pet2anat_method,
            crop_anat=False,
            sloppy=self.inputs.sloppy,
            name='pet_reg_uncropped_fallback_wf',
        )
        fallback_wf.base_dir = os.path.abspath(cwd)
        fallback_wf.inputs.inputnode.ref_pet_brain = self.inputs.ref_pet_brain
        fallback_wf.inputs.inputnode.anat_preproc = self.inputs.anat_preproc
        fallback_wf.inputs.inputnode.anat_mask = self.inputs.anat_mask

        fallback_wf.run(plugin='Linear')
        fallback_dir = Path(fallback_wf.base_dir) / fallback_wf.name

        if self.inputs.pet2anat_method == 'auto':
            outputs = loadpkl(
                str(fallback_dir / 'select_best' / 'result_select_best.pklz')
            ).outputs
            return outputs.best_xfm, outputs.best_inv_xfm, outputs.winner, outputs.best_score

        xfm_outputs = loadpkl(
            str(fallback_dir / 'convert_xfm' / 'result_convert_xfm.pklz')
        ).outputs
        score_outputs = loadpkl(
            str(fallback_dir / 'score_registration' / 'result_score_registration.pklz')
        ).outputs
        return xfm_outputs.out_xfm, xfm_outputs.out_inv, None, score_outputs.similarity

    def _set_results(self, xfm, inv_xfm, winner, score, *, fallback, anat_reference):
        self._results['best_transform'] = xfm
        self._results['best_inv_transform'] = inv_xfm
        self._results['best_winner'] = winner
        self._results['best_score'] = score
        self._results['fallback'] = fallback
        self._results['anat_reference'] = anat_reference
        self._results['registration_winner'] = winner
        self._results['registration_score'] = score

    def _require_defined(self, *names):
        undefined = [name for name in names if not isdefined(getattr(self.inputs, name))]
        if undefined:
            raise ValueError(
                'Missing inputs required for automatic PET-to-anatomical fallback: '
                + ', '.join(undefined)
            )
