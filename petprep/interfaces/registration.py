# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Registration helper interfaces."""

from pathlib import Path

from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    isdefined,
    traits,
)


class PETCoregistrationFallbackInputSpec(BaseInterfaceInputSpec):
    interface_version = traits.Str('6', usedefault=True)
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
                score is not None and score <= self.inputs.fallback_threshold
                for score in (self.inputs.cropped_ants_score, self.inputs.cropped_fs_score)
            )
        else:
            cropped = self._select_cropped_manual()
            self._score_summary['cropped'] = {
                self.inputs.pet2anat_method: self.inputs.cropped_score,
                'winner': cropped[2],
                'score': cropped[3],
            }
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

    def _ensure_inverse_transform(self, xfm, inv_xfm):
        if isdefined(inv_xfm) and Path(inv_xfm).exists():
            return inv_xfm

        import nitransforms as nt

        inv_xfm = Path(self._runtime_cwd) / 'out_inv.tfm'
        (~nt.linear.load(xfm, fmt='itk')).to_filename(inv_xfm, fmt='itk')
        return str(inv_xfm)

    def _set_results(self, xfm, inv_xfm, winner, score, *, fallback, anat_reference):
        self._require_selected_outputs(xfm, inv_xfm, score)
        score = float(score)
        winner = str(winner) if winner is not None and isdefined(winner) else None
        if fallback:
            xfm, inv_xfm = self._copy_selected_transforms(xfm, inv_xfm)

        self._results['best_transform'] = xfm
        self._results['best_inv_transform'] = inv_xfm
        self._results['best_winner'] = winner
        self._results['best_score'] = score
        self._results['fallback'] = fallback
        self._results['anat_reference'] = anat_reference
        self._results['registration_winner'] = winner
        self._results['registration_score'] = score
        self._results['fallback_scores'] = self._write_score_summary(
            winner=winner,
            score=score,
            fallback=fallback,
            anat_reference=anat_reference,
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

    def _write_score_summary(self, *, winner, score, fallback, anat_reference):
        import json

        summary_file = Path(self._runtime_cwd) / 'fallback_scores.json'
        self._score_summary['selected'] = {
            'winner': winner,
            'score': score,
            'fallback': fallback,
            'anat_reference': anat_reference,
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
