import numpy as np
import numpy.typing as npt


class Mixin:
    def __init__(self):

        timeConstants = self.get_TimeConstants(
            f=self.f_c[:, 0], tau100_s=0.030, tauMin_s=0.008
        ).reshape(1, self.numBarkBands)

        self.patternProcessingAlpha: npt.ArrayLike = np.exp(
            -1 / (self.Fss_fft * timeConstants)
        )

    def excitationPatternProcessing(
        self,
        EsTilde_T: npt.ArrayLike,
        EsTilde_R: npt.ArrayLike,
    ):
        numChannels, numBands, numFrames = EsTilde_R.shape

        EsTilde_R = self.AR_filter(X=EsTilde_R, alpha=self.patternProcessingAlpha)

        EsTilde_T = self.AR_filter(X=EsTilde_T, alpha=self.patternProcessingAlpha)

        momentaryCorrection = np.sum(np.sqrt(EsTilde_T * EsTilde_R), axis=1)
        momentaryCorrection = momentaryCorrection / np.sum(EsTilde_T, axis=1)
        momentaryCorrection = np.square(momentaryCorrection).reshape(
            numChannels, 1, numFrames
        )

        idx = np.where(momentaryCorrection < 1)
        # E_{LR}, E_{LT}
        patternMomentaryCorrected_ref = EsTilde_R
        patternMomentaryCorrected_ref[idx] = EsTilde_R[idx] / momentaryCorrection[idx]

        patternMomentaryCorrected_test = EsTilde_T * momentaryCorrection
        patternMomentaryCorrected_test[idx] = EsTilde_T[idx]

        correlation_num = self.AR_filter(
            X=patternMomentaryCorrected_ref * patternMomentaryCorrected_test,
            alpha=self.patternProcessingAlpha,
        )
        correlation_den = self.AR_filter(
            X=patternMomentaryCorrected_ref * patternMomentaryCorrected_ref,
            alpha=self.patternProcessingAlpha,
        )

        correlation_ref = np.ones_like(correlation_num)
        correlation_test = np.ones_like(correlation_den)

        cond = np.asarray(correlation_num < correlation_den)

        idx = cond.nonzero()
        correlation_ref[idx] = correlation_num[idx] / correlation_den[idx]

        idx = (1 - cond).nonzero()
        correlation_test[idx] = correlation_den[idx] / correlation_num[idx]

        correlation_ref[:, 0, :] = 1.0
        correlation_test[:, 0, :] = 1.0

        correlationFreqSmooth_ref = self.frequencySmoothing(correlation_ref)
        correlationFreqSmooth_test = self.frequencySmoothing(correlation_test)

        # P_{CR}, P_{CT}
        patternCorrectionFactor_ref = self.AR_filter(
            X=correlationFreqSmooth_ref, alpha=self.patternProcessingAlpha
        )
        patternCorrectionFactor_test = self.AR_filter(
            X=correlationFreqSmooth_test, alpha=self.patternProcessingAlpha
        )

        # E_{PR}, E_{PT}
        EP_R = patternMomentaryCorrected_ref / patternCorrectionFactor_ref

        EP_T = patternMomentaryCorrected_test / patternCorrectionFactor_test

        return EP_T, EP_R

    def modulationPatternProcessing(
        self,
        Es_T: npt.ArrayLike,
        Es_R: npt.ArrayLike,
    ) -> tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]:

        # Average loudness, Eq. (65)
        Ebar_R: npt.ArrayLike = self.AR_filter(
            X=np.power(Es_R, 0.3), alpha=self.timeToFreqAlpha
        )
        Ebar_T: npt.ArrayLike = self.AR_filter(
            X=np.power(Es_T, 0.3), alpha=self.timeToFreqAlpha
        )

        Es_T_prev = np.zeros_like(Es_T)
        Es_T_prev[:, :, 1:] = Es_T[:, :, :-1]

        Es_R_prev = np.zeros_like(Es_R)
        Es_R_prev[:, :, 1:] = Es_R[:, :, :-1]

        # Average loudness difference, Eq. (66)
        Dbar_R = self.AR_filter(
            X=np.abs(np.power(Es_R, 0.3) - np.power(Es_R_prev, 0.3)) * self.Fss_fft,
            alpha=self.timeToFreqAlpha,
        )
        Dbar_T = self.AR_filter(
            X=np.abs(np.power(Es_T, 0.3) - np.power(Es_T_prev, 0.3)) * self.Fss_fft,
            alpha=self.timeToFreqAlpha,
        )

        # MOdulation parameters, Eq. (67)
        M_T: npt.ArrayLike = Dbar_T / (1 + Ebar_T / 0.3)
        M_R: npt.ArrayLike = Dbar_R / (1 + Ebar_R / 0.3)

    
        return M_T, M_R, Ebar_R

    def loudnessCalculation(
        self,
        EsTilde_T: npt.ArrayLike,
        EsTilde_R: npt.ArrayLike,
    ) -> tuple[npt.ArrayLike, npt.ArrayLike]:

        c: float = 1.07664
        E0: float = 1e4

        commonFactor = c * np.power(self.Et / (self.s * E0), 0.23).reshape(
            1, self.numBarkBands, 1
        )

        # N_R, N_T
        N_T = np.power(1 - self.s + self.s * EsTilde_T / self.Et, 0.23) - 1
        N_T = N_T * commonFactor

        N_R = np.power(1 - self.s + self.s * EsTilde_R / self.Et, 0.23) - 1
        N_R = N_R * commonFactor

        Ntot_T: npt.ArrayLike = (
            np.sum(np.maximum(N_T, 0), axis=1) * 24 / self.numBarkBands
        )
        Ntot_R: npt.ArrayLike = (
            np.sum(np.maximum(N_R, 0), axis=1) * 24 / self.numBarkBands
        )
        return Ntot_T, Ntot_R
