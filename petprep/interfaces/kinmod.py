import numpy as np
import pandas as pd
from scipy.stats import linregress
from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec, TraitedSpec, File, traits

class LoganModelInputSpec(BaseInterfaceInputSpec):
    reference_tac = traits.Array(desc="Reference tissue TAC", mandatory=True)
    target_tac = traits.Array(desc="Target tissue TAC", mandatory=True)
    frame_times = traits.Array(desc="Frame times", mandatory=True)
    start_time = traits.Float(desc="Start time", mandatory=True)

class LoganModelOutputSpec(TraitedSpec):
    Kappa2 = traits.Float(desc="Kappa2 value")
    VT = traits.Float(desc="VT value")

class LoganModel(BaseInterface):
    input_spec = LoganModelInputSpec
    output_spec = LoganModelOutputSpec

    @staticmethod
    def cumtrapz_l(t, y):
        return np.cumsum(y * np.diff(t, prepend=0))

    def _run_interface(self, runtime):
        reference_tac = self.inputs.reference_tac
        target_tac = self.inputs.target_tac
        frame_times = self.inputs.frame_times
        start_time = self.inputs.start_time

        start_idx = np.argmax(frame_times >= start_time)

        ref_integral = self.cumtrapz_l(frame_times, reference_tac)
        tar_integral = self.cumtrapz_l(frame_times, target_tac)

        x = ref_integral[start_idx:] / target_tac[start_idx:]
        y = tar_integral[start_idx:] / target_tac[start_idx:]

        # Linear regression
        slope, intercept, r_value, p_value, std_err = linregress(x, y)

        self._Kappa2 = -1 / intercept
        self._VT = slope

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["Kappa2"] = self._Kappa2
        outputs["VT"] = self._VT
        return outputs