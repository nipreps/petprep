# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
from json import loads

from niworkflows.interfaces.bids import DerivativesDataSink as _DDSink

from petprep.utils.bids import load_data

from .cifti import GeneratePetCifti
from .motion import MotionPlot
from .reports import AtlasROIsReport
from .tacs import ExtractRefTAC, ExtractTACs


class DerivativesDataSink(_DDSink):
    out_path_base = ''
    _petprep_spec = loads(load_data.readable('nipreps.json').read_text())
    _config_entities = frozenset({e['name'] for e in _petprep_spec['entities']})
    _config_entities_dict = _petprep_spec['entities']
    _file_patterns = tuple(_petprep_spec['default_path_patterns'])


__all__ = (
    'DerivativesDataSink',
    'GeneratePetCifti',
    'ExtractTACs',
    'ExtractRefTAC',
    'MotionPlot',
    'AtlasROIsReport',
)
