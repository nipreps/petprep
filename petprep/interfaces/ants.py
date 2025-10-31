from __future__ import annotations

import time

from nipype.interfaces.ants import Registration
from nipype.interfaces.base import TraitedSpec, traits


try:  # Newer Nipype exposes a private ``_output_spec`` attribute
    _BaseOutputSpec = Registration._output_spec
except AttributeError:  # pragma: no cover - older releases only define ``output_spec``
    _BaseOutputSpec = getattr(Registration, 'output_spec', TraitedSpec)
    if not isinstance(_BaseOutputSpec, type):
        _BaseOutputSpec = _BaseOutputSpec.__class__


class _TimedRegistrationOutputSpec(_BaseOutputSpec):
    runtime_seconds = traits.Float(desc='Elapsed wall-clock time for the registration stage.')


class TimedRegistration(Registration):
    """ANTs Registration interface that tracks wall-clock execution time."""

    output_spec = _TimedRegistrationOutputSpec
    _output_spec = _TimedRegistrationOutputSpec

    def _run_interface(self, runtime):
        start = time.time()
        runtime = super()._run_interface(runtime)
        self._runtime_seconds = time.time() - start
        return runtime

    def _list_outputs(self):
        outputs = super()._list_outputs()
        outputs['runtime_seconds'] = getattr(self, '_runtime_seconds', None)
        return outputs


__all__ = ('TimedRegistration',)
