import numpy as np
import numpy.typing as npt


class Mixin:
    def __init__(self):

        timeConstants = self.get_TimeConstants(
            f=self.f_c[:, 0], tau100_s=0.050, tauMin_s=0.008
        ).reshape(1, self.numBarkBands)

        self.patternProcessingAlpha: npt.ArrayLike = np.exp(
            -1 / (self.Fss_fft * timeConstants)
        )

        # s
        self.s = self.get_thresholdIndex(f=self.f_c).reshape(1, self.numBarkBands, 1)
        # E_t
        self.Et = self.get_excitationThreshold(f=self.f_c).reshape(
            1, self.numBarkBands, 1
        )

    def get_thresholdIndex(self, f):
        # Threshold index, eq. (69)
        s_dB = (
            -2
            - 2.05 * np.arctan(f * 0.00025)
            - 0.75 * np.arctan(np.square(f * 0.000625))
        )
        s = self.idB10(s_dB)
        return s

    def get_excitationThreshold(self, f):
        # Excitation threshold, eq. (70)
        Et_dB = 3.64 * np.power((f / 1000), -0.8)
        Et = self.idB10(Et_dB)
        return Et

    def frequencySmoothing(self, R: npt.ArrayLike):
        numChannels, numBands, numFrames = R.shape

        Ra = np.zeros_like(R)

        for k in range(numBands):
            # Eq. (63)
            M1 = min(self.FFTM1, k)
            M2 = min(self.FFTM2, self.numBarkBands - 1 - k)
            # Frequency smoothed terms, Eq. (62)
            Ra[:, k, :] = np.sum(R[:, k - M1 : k + M2 + 1, :], axis=1) / (M1 + M2 + 1)
            # for i in range(k - M1, k + M2 + 1):
            #     Ra[:, k, :] += R[:, i, :]

        return Ra

    def excitationPatternProcessing(
        self,
        EsTilde_T: npt.ArrayLike,
        EsTilde_R: npt.ArrayLike,
    ):
        numChannels, numBands, numFrames = EsTilde_R.shape

        # Time spreading, Eq. (56)
        P_R = self.AR_filter(X=EsTilde_R, alpha=self.patternProcessingAlpha)
        P_T = self.AR_filter(X=EsTilde_T, alpha=self.patternProcessingAlpha)

        # Momentary correction factor, Eq. (57)
        C_L = np.sum(np.sqrt(P_T * P_R), axis=1, keepdims=True)
        C_L /= np.sum(P_T, axis=1, keepdims=True)
        C_L = np.square(C_L)

        # Correcting the excitation patterns, Eq. (58)
        EL_R = np.where(C_L > 1, EsTilde_R / C_L, EsTilde_R)
        EL_T = np.where(C_L > 1, EsTilde_T, EsTilde_T * C_L)

        # Time smoothed correlation, Eq. (59)
        Rn = self.AR_filter(
            X=EL_T * EL_R,
            alpha=self.patternProcessingAlpha,
        )
        Rd = self.AR_filter(
            X=EL_R * EL_R,
            alpha=self.patternProcessingAlpha,
        )

        # Auxiliary signals, Eq. (60)
        R_R = np.where(Rn >= Rd, 1, Rn / Rd)
        R_T = np.where(Rn >= Rd, Rd / Rn, 1)

        # Frequency smoothing, Eq. (62)
        Ra_R = self.frequencySmoothing(R_R)
        Ra_T = self.frequencySmoothing(R_T)

        # Pattern correction factors, Eq. (61)
        PC_R = self.AR_filter(X=Ra_R, alpha=self.patternProcessingAlpha)
        PC_T = self.AR_filter(X=Ra_T, alpha=self.patternProcessingAlpha)

        # Spectrally adapted patterns, Eq. (64)
        EP_T = EL_T * PC_T
        EP_R = EL_R * PC_R

        return EP_T, EP_R

    def modulationPatternProcessing(
        self,
        Es_T: npt.ArrayLike,
        Es_R: npt.ArrayLike,
    ) -> tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]:

        # Average loudness, Eq. (65)
        Ebar_R: npt.ArrayLike = self.AR_filter(
            X=np.power(Es_R, 0.3), alpha=self.patternProcessingAlpha
        )
        Ebar_T: npt.ArrayLike = self.AR_filter(
            X=np.power(Es_T, 0.3), alpha=self.patternProcessingAlpha
        )

        Es_T_prev = np.zeros_like(Es_T)
        Es_T_prev[:, :, 1:] = Es_T[:, :, :-1]

        Es_R_prev = np.zeros_like(Es_R)
        Es_R_prev[:, :, 1:] = Es_R[:, :, :-1]

        # Average loudness difference, Eq. (66)
        Dbar_R = self.AR_filter(
            X=np.abs(np.power(Es_R, 0.3) - np.power(Es_R_prev, 0.3)) * self.Fss_fft,
            alpha=self.patternProcessingAlpha,
        )
        Dbar_T = self.AR_filter(
            X=np.abs(np.power(Es_T, 0.3) - np.power(Es_T_prev, 0.3)) * self.Fss_fft,
            alpha=self.patternProcessingAlpha,
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

        # Specific loudness patterns, Eq. (68)
        commonFactor = c * np.power(self.Et / (self.s * E0), 0.23).reshape(
            1, self.numBarkBands, 1
        )

        N_T = np.power(1 - self.s + self.s * EsTilde_T / self.Et, 0.23) - 1
        N_T *= commonFactor

        N_R = np.power(1 - self.s + self.s * EsTilde_R / self.Et, 0.23) - 1
        N_R *= commonFactor

        # Total loudness, Eq. (71)
        Ntot_T: npt.ArrayLike = np.mean(np.maximum(N_T, 0), axis=1) * 24
        Ntot_R: npt.ArrayLike = np.mean(np.maximum(N_R, 0), axis=1) * 24
        return Ntot_T, Ntot_R
