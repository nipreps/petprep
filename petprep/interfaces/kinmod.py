import numpy as np
from scipy import integrate
from scipy.optimize import curve_fit
import statsmodels.api as sm
from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec, TraitedSpec, File, traits

class LoganModelInputSpec(BaseInterfaceInputSpec):
    frame_times = traits.Array(desc="Mid-time for each frame (in minutes)", mandatory=True)
    reference_tac = traits.Array(desc="Reference tissue TAC", mandatory=True)
    target_tac = traits.Array(desc="ROI tissue TAC", mandatory=True)
    tstar = traits.Float(desc="Start time for linear regression", mandatory=True)
    k2ref = traits.Float(desc="k2ref", mandatory=True)
    frame_weight = traits.Array(desc="Weight of frame when fitting", mandatory=True)


class LoganModelOutputSpec(TraitedSpec):
    Kappa2 = traits.Float(desc="Kappa2 value")
    BPnd = traits.Float(desc="BPnd value")
    MSE = traits.Float(desc="Mean Squared Error")
    FPE = traits.Float(desc="Final Prediction Error")
    SigmaSqr = traits.Float(desc="Squared Standard Deviation of Prediction Error")
    LogLike = traits.Float(desc="Log-likelihood")
    AIC = traits.Float(desc="Akaike Information Criterion")

class LoganModel(BaseInterface):
    input_spec = LoganModelInputSpec
    output_spec = LoganModelOutputSpec

    def _run_interface(self, runtime):
        frame_times = self.inputs.frame_times
        reference_tac = self.inputs.reference_tac
        target_tac = self.inputs.target_tac
        tstar = self.inputs.tstar
        k2ref = self.inputs.k2ref
        frame_weight = self.inputs.frame_weight

        start_idx = np.where(frame_times > tstar)[0][0]

        cref_int = integrate.cumtrapz(reference_tac, frame_times, initial=0)
        croi_int = integrate.cumtrapz(target_tac, frame_times, initial=0)

        x = (cref_int + reference_tac / k2ref) / target_tac
        y = croi_int / target_tac

        xfit = x[start_idx:]
        yfit = y[start_idx:]
        frame_weights_fit = frame_weight[start_idx:]

        xfit_add_constant = sm.add_constant(xfit)

        glm = sm.GLM(yfit, xfit_add_constant, family=sm.families.Gaussian(), var_weights=frame_weights_fit)

        fitted_glm = glm.fit()

        yhat = fitted_glm.fittedvalues

        self._Kappa2 = -1 / fitted_glm.params[0]
        self._BPnd = fitted_glm.params[1] - 1

        self._MSE = np.sum((yhat - yfit) ** 2) / (len(yfit) - 3)
        self._FPE = np.sum((yhat - yfit) ** 2) * (len(yfit) + 3) / (len(yfit) - 3)
        self._SigmaSqr = np.std(yhat - yfit) ** 2
        self._LogLike = -0.5 * len(yfit) * np.log(2 * np.pi * self._SigmaSqr) - 0.5 * np.sum((yhat - yfit) ** 2) / self._SigmaSqr
        self._AIC = -2 * self._LogLike + 2 * 4

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["Kappa2"] = self._Kappa2
        outputs["BPnd"] = self._BPnd
        outputs["MSE"] = self._MSE
        outputs["FPE"] = self._FPE
        outputs["SigmaSqr"] = self._SigmaSqr
        outputs["LogLike"] = self._LogLike
        outputs["AIC"] = self._AIC
        return outputs

class MRTM2InputSpec(BaseInterfaceInputSpec):
    frame_times = traits.Array(desc="Mid-time for each frame (in minutes)", mandatory=True)
    frame_weight = traits.Array(desc="Weight of frame when fitting", mandatory=True)
    reference_tac = traits.Array(desc="Reference tissue TAC", mandatory=True)
    target_tac = traits.Array(desc="ROI tissue TAC", mandatory=True)
    k2p = traits.Float(desc="k2 prime used when estimating BPnd and R1 for all regions", mandatory=True)

class MRTM2OutputSpec(TraitedSpec):
    R1 = traits.Float(desc="R1 value")
    BPnd = traits.Float(desc="BPnd value")
    MSE = traits.Float(desc="Mean Squared Error")
    FPE = traits.Float(desc="Final Prediction Error")
    SigmaSqr = traits.Float(desc="Squared Standard Deviation of Prediction Error")
    LogLike = traits.Float(desc="Log-likelihood")
    AIC = traits.Float(desc="Akaike Information Criterion")

class MRTM2(BaseInterface):
    input_spec = MRTM2InputSpec
    output_spec = MRTM2OutputSpec

    def _run_interface(self, runtime):
        frame_times = self.inputs.frame_times
        reference_tac = self.inputs.reference_tac
        target_tac = self.inputs.target_tac
        k2p = self.inputs.k2p
        frame_weight = self.inputs.frame_weight

        cref_int = integrate.cumtrapz(reference_tac, frame_times, initial=0) + (1/k2p) * reference_tac
        croi_int = integrate.cumtrapz(target_tac, frame_times, initial=0)

        x = np.column_stack((cref_int, croi_int))
        y = target_tac[:,np.newaxis]

        glm = sm.GLM(y, x, family=sm.families.Gaussian(), var_weights=frame_weight)

        fitted_glm = glm.fit()

        yhat = fitted_glm.fittedvalues

        self._R1 = fitted_glm.params[0] / k2p
        self._BPnd = -(fitted_glm.params[0] / fitted_glm.params[1] + 1)

        self._MSE = np.sum((yhat - y) ** 2) / (len(y) - 3)
        self._FPE = np.sum((yhat - y) ** 2) * (len(y) + 3) / (len(y) - 3)
        self._SigmaSqr = np.std(yhat - y) ** 2
        self._LogLike = -0.5 * len(y) * np.log(2 * np.pi * self._SigmaSqr) - 0.5 * np.sum((yhat - y) ** 2) / self._SigmaSqr
        self._AIC = -2 * self._LogLike + 2 * 4

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["R1"] = self._R1
        outputs["BPnd"] = self._BPnd
        outputs["MSE"] = self._MSE
        outputs["FPE"] = self._FPE
        outputs["SigmaSqr"] = self._SigmaSqr
        outputs["LogLike"] = self._LogLike
        outputs["AIC"] = self._AIC
        return outputs
