# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Registration helper interfaces."""

from pathlib import Path

import numpy as np
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
    anatref_strategy = traits.Enum('t1w', 'nu', value='t1w', usedefault=True)

    cropped_transform = File(exists=True)
    cropped_inv_transform = File(exists=True)
    cropped_winner = traits.Either(None, traits.Str(), usedefault=True)
    cropped_score = traits.Either(None, traits.Float(), usedefault=True)
    cropped_anat_reference = traits.Enum('cropped', 'uncropped')
    cropped_reference_policy = traits.Str()

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
    reference_policy = traits.Str()
    registration_winner = traits.Either(None, traits.Str())
    registration_score = traits.Either(None, traits.Float())
    fallback_scores = File(exists=True)


class PETCoregistrationFallback(SimpleInterface):
    """Select cropped registration or lazily run an uncropped fallback."""

    input_spec = PETCoregistrationFallbackInputSpec
    output_spec = PETCoregistrationFallbackOutputSpec

    def _run_interface(self, runtime):
        self._runtime_cwd = runtime.cwd
        self._score_summary = {
            'threshold': self.inputs.fallback_threshold,
            'pet2anat_method': self.inputs.pet2anat_method,
            'anatref_strategy': self.inputs.anatref_strategy,
            'cropped': {},
            'uncropped': {},
        }
        if self.inputs.pet2anat_method == 'auto':
            cropped = self._select_cropped_auto()
            self._score_summary['cropped'] = {
                'ants': self.inputs.cropped_ants_score,
                'freesurfer': self.inputs.cropped_fs_score,
                'winner': cropped[2],
                'score': cropped[3],
            }
            should_fallback = not any(
                self._score_passes_threshold(score)
                for score in (self.inputs.cropped_ants_score, self.inputs.cropped_fs_score)
            )
        else:
            cropped = self._select_cropped_manual()
            self._score_summary['cropped'] = {
                self.inputs.pet2anat_method: self.inputs.cropped_score,
                'winner': cropped[2],
                'score': cropped[3],
            }
            should_fallback = not self._score_passes_threshold(self.inputs.cropped_score)

        cropped_anat_reference, cropped_reference_policy = self._reference_metadata(
            cropped[2],
            crop_anat=True,
        )
        if isdefined(self.inputs.cropped_anat_reference):
            cropped_anat_reference = self.inputs.cropped_anat_reference
        if isdefined(self.inputs.cropped_reference_policy):
            cropped_reference_policy = self.inputs.cropped_reference_policy

        # The strict PETSurfer-style nu.mgz route is already uncropped, so rerunning
        # the same manual mri_coreg workflow cannot provide a different fallback.
        if self.inputs.anatref_strategy == 'nu' and self.inputs.pet2anat_method == 'mri_coreg':
            should_fallback = False

        if not should_fallback:
            self._set_results(
                *cropped,
                fallback=False,
                anat_reference=cropped_anat_reference,
                reference_policy=cropped_reference_policy,
            )
            return runtime

        self._require_defined('ref_pet_brain', 'anat_preproc', 'anat_mask')
        uncropped = self._run_uncropped_fallback(runtime.cwd)
        uncropped_anat_reference, uncropped_reference_policy = self._reference_metadata(
            uncropped[2],
            crop_anat=False,
        )
        if uncropped[3] is not None and (cropped[3] is None or uncropped[3] < cropped[3]):
            self._set_results(
                *uncropped,
                fallback=True,
                anat_reference=uncropped_anat_reference,
                reference_policy=uncropped_reference_policy,
            )
        else:
            self._set_results(
                *cropped,
                fallback=False,
                anat_reference=cropped_anat_reference,
                reference_policy=cropped_reference_policy,
            )

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

        from nipype.utils.filemanip import loadpkl

        from petprep.workflows.pet.registration import init_pet_reg_wf

        if not hasattr(self, '_runtime_cwd'):
            self._runtime_cwd = os.path.abspath(cwd)
        if not hasattr(self, '_score_summary'):
            self._score_summary = {'uncropped': {}}

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
        fallback_wf.inputs.inputnode.anatref_strategy = self.inputs.anatref_strategy

        fallback_wf.run(plugin='Linear')
        fallback_dir = Path(fallback_wf.base_dir) / fallback_wf.name

        if self.inputs.pet2anat_method == 'auto':
            from petprep.workflows.pet.registration import _select_best_transform

            ants_outputs = loadpkl(
                str(fallback_dir / 'convert_xfm_ants' / 'result_convert_xfm_ants.pklz')
            ).outputs
            fs_outputs = loadpkl(
                str(fallback_dir / 'convert_xfm_fs' / 'result_convert_xfm_fs.pklz')
            ).outputs
            ants_score = loadpkl(
                str(fallback_dir / 'score_ants' / 'result_score_ants.pklz')
            ).outputs.similarity
            fs_score = loadpkl(
                str(fallback_dir / 'score_fs' / 'result_score_fs.pklz')
            ).outputs.similarity
            xfm, inv_xfm, winner, score = _select_best_transform(
                ants_outputs.out_xfm,
                fs_outputs.out_xfm,
                ants_outputs.out_inv,
                fs_outputs.out_inv,
                ants_score,
                fs_score,
            )
            self._score_summary['uncropped'] = {
                'ants': ants_score,
                'freesurfer': fs_score,
                'winner': winner,
                'score': score,
            }
            return xfm, self._ensure_inverse_transform(xfm, inv_xfm), winner, score

        xfm_outputs = loadpkl(
            str(fallback_dir / 'convert_xfm' / 'result_convert_xfm.pklz')
        ).outputs
        score_outputs = loadpkl(
            str(fallback_dir / 'score_registration' / 'result_score_registration.pklz')
        ).outputs
        inv_xfm = self._ensure_inverse_transform(xfm_outputs.out_xfm, xfm_outputs.out_inv)
        self._score_summary['uncropped'] = {
            self.inputs.pet2anat_method: score_outputs.similarity,
            'winner': None,
            'score': score_outputs.similarity,
        }
        return xfm_outputs.out_xfm, inv_xfm, None, score_outputs.similarity

    def _reference_metadata(self, winner, *, crop_anat):
        from petprep.workflows.pet.registration import _describe_registration_reference

        registration_method = (
            str(winner)
            if winner is not None and isdefined(winner)
            else self.inputs.pet2anat_method
        )
        return _describe_registration_reference(
            self.inputs.anatref_strategy,
            registration_method,
            crop_anat,
        )

    def _score_passes_threshold(self, score):
        """Return whether a rounded score is strictly better than the threshold."""
        score = self._round_score(score)
        threshold = self._round_score(self.inputs.fallback_threshold)
        return score is not None and score < threshold

    def _round_score(self, score):
        if score is None or not isdefined(score):
            return None

        precision = max(_decimal_places(self.inputs.fallback_threshold), 0)
        rounded = float(np.round(float(score), decimals=precision))
        return 0.0 if rounded == 0 else rounded

    def _ensure_inverse_transform(self, xfm, inv_xfm):
        if isdefined(inv_xfm) and Path(inv_xfm).exists():
            return inv_xfm

        import nitransforms as nt

        inv_xfm = Path(self._runtime_cwd) / 'out_inv.tfm'
        (~nt.linear.load(xfm, fmt='itk')).to_filename(inv_xfm, fmt='itk')
        return str(inv_xfm)

    def _set_results(
        self,
        xfm,
        inv_xfm,
        winner,
        score,
        *,
        fallback,
        anat_reference,
        reference_policy,
    ):
        self._require_selected_outputs(xfm, inv_xfm, score)
        score = self._round_score(score)
        winner = str(winner) if winner is not None and isdefined(winner) else None
        if fallback:
            xfm, inv_xfm = self._copy_selected_transforms(xfm, inv_xfm)

        self._results['best_transform'] = xfm
        self._results['best_inv_transform'] = inv_xfm
        self._results['best_winner'] = winner
        self._results['best_score'] = score
        self._results['fallback'] = fallback
        self._results['anat_reference'] = anat_reference
        self._results['reference_policy'] = reference_policy
        self._results['registration_winner'] = winner
        self._results['registration_score'] = score
        self._results['fallback_scores'] = self._write_score_summary(
            winner=winner,
            score=score,
            fallback=fallback,
            anat_reference=anat_reference,
            reference_policy=reference_policy,
        )

    def _copy_selected_transforms(self, xfm, inv_xfm):
        import shutil

        def _copy_transform(src, name):
            src = Path(src)
            suffix = src.suffix or '.tfm'
            dst = Path(self._runtime_cwd) / f'{name}{suffix}'
            if src.resolve() != dst.resolve():
                shutil.copyfile(src, dst)
            return str(dst)

        return (
            _copy_transform(xfm, 'best_transform'),
            _copy_transform(inv_xfm, 'best_inv_transform'),
        )

    def _write_score_summary(
        self,
        *,
        winner,
        score,
        fallback,
        anat_reference,
        reference_policy,
    ):
        import json

        summary_file = Path(self._runtime_cwd) / 'fallback_scores.json'
        for section in ('cropped', 'uncropped'):
            self._score_summary[section] = {
                key: self._round_score(value) if key not in ('winner',) else value
                for key, value in self._score_summary.get(section, {}).items()
            }
        self._score_summary['selected'] = {
            'winner': winner,
            'score': score,
            'fallback': fallback,
            'anat_reference': anat_reference,
            'reference_policy': reference_policy,
        }
        summary_file.write_text(json.dumps(self._score_summary, indent=2, sort_keys=True))
        return str(summary_file)

    def _require_selected_outputs(self, xfm, inv_xfm, score):
        missing = []
        if not isdefined(xfm):
            missing.append('best_transform')
        if not isdefined(inv_xfm):
            missing.append('best_inv_transform')
        if score is None or not isdefined(score):
            missing.append('best_score')
        if missing:
            raise ValueError(
                'PET-to-anatomical fallback selected an incomplete registration result: '
                + ', '.join(missing)
            )

    def _require_defined(self, *names):
        undefined = [name for name in names if not isdefined(getattr(self.inputs, name))]
        if undefined:
            raise ValueError(
                'Missing inputs required for automatic PET-to-anatomical fallback: '
                + ', '.join(undefined)
            )


def _decimal_places(value):
    """Infer the meaningful decimal precision from a numeric threshold."""
    value = str(float(value)).rstrip('0').rstrip('.')
    return len(value.rpartition('.')[2]) if '.' in value else 0
