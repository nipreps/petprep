"""Derivative naming for PET-only atlas outputs."""

from . import DerivativesDataSink

_ACQUISITION = (
    'sub-{subject}[_ses-{session}][_task-{task}][_acq-{acquisition}]'
    '[_ce-{ceagent}][_trc-{tracer}][_rec-{reconstruction}][_run-{run}]'
)
_PET_PREFIX = 'sub-{subject}[/ses-{session}]/{datatype<pet>|pet}/' + _ACQUISITION


class PETOnlyDataSink(DerivativesDataSink):
    """Keep atlas tables, TACs, and transforms distinct across output resolutions.

    This extends NiPreps patterns locally to the PET-only branch. In particular,
    PET-space dseg tables/images and resolution-specific TACs/transforms are not
    all represented by the upstream patterns. ``seg`` remains PETPrep's custom
    entity, inserted by the parent sink when requested.
    """

    _file_patterns = (
        _PET_PREFIX + '[_space-{space}][_cohort-{cohort}][_res-{resolution}]'
        '[_label-{label}][_desc-{desc}]_{suffix<pet|petref|dseg|morph|tacs|mask|probseg>}'
        '{extension<.nii|.nii.gz|.tsv|.json>}',
        _PET_PREFIX + '[_res-{resolution}]_from-{from}_to-{to}_mode-{mode<image|points>|image}_'
        '{suffix<xfm>}{extension<.txt|.h5>}',
        'sub-{subject}/{datatype<figures>}/'
        + _ACQUISITION
        + '[_space-{space}][_cohort-{cohort}][_res-{resolution}][_desc-{desc}]_'
        '{suffix<pet>}{extension<.html|.svg>}',
        *DerivativesDataSink._file_patterns,
    )
