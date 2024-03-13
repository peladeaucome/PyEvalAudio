import numpy as np
import numpy.typing as npt
from . import critical_bands
import matplotlib.pyplot as plt

from numba import njit


class Mixin:
    def __init__(self, mode="basic", Amax=32768):

        self.NF = 2048  # Window size
        self.hopSize = 1024  # Frame advance
        self.sr_hz = 48000  # Sample rate

        self.mode = mode
        if mode == "basic":
            self.numBarkBands = 109
            self.barkWidth = 0.25
            self.FFTM1, self.FFTM2 = 3, 4
        elif mode == "advanced":
            self.numBarkBands = 55
            self.barkWidth = 0.5
            self.FFTM1, self.FFTM2 = (1, 2)
            self.FBM1, self.FBM2 = (1, 1)
            raise NotImplementedError("The advanced mode has not been implemented yet")

        # A_max : maximum value that a file can have
        self.Amax: float = Amax

        # FFT frequencies
        self.numFftBands: int = 1025
        self.f_hz: npt.ArrayLike = np.fft.rfftfreq(n=2048, d=1 / self.sr_hz)

        # loudness scaling
        self.G_L: float = 3.503992537617512
        # gamma = 0.8497
        # Lp = 92
        # self.G_L = 1000
        # self.G_L = (
        #     self.idB20(Lp)
        #     / (gamma * (self.Amax / 4) * (self.NF - 1))
        #     * np.sqrt(3 / 8)
        #     * self.NF
        # )

        # Hann window
        self.window: npt.ArrayLike = self.get_hannWindow(NF=self.NF)

        # FFT weighting filter
        self.W: npt.ArrayLike = self.get_earFilter()

        # Critical bands of the FFT model
        self.f_l, self.f_c, self.f_u = self.get_BarkBandsFreqs()

        # Grouping matrix
        self.U = self.get_barkBinGrouping()

        # Internal noise
        self.E_IN = self.get_internalNoise(f=self.f_c).reshape(1, self.numBarkBands, 1)

        # Precompute frequency spreading normalization factor
        self.B_s = 1
        self.B_s = self.frequencySpreading(
            E=np.ones((1, self.numBarkBands, 1))
        )  # .reshape(1, self.numBarkBands)

        # Precompute time spreading factors
        self.Fss_fft = self.sr_hz * 2 / self.NF

        timeConstants = self.get_TimeConstants(
            f=self.f_c[:, 0], tau100_s=0.030, tauMin_s=0.008
        ).reshape(1, self.numBarkBands)
        self.timeToFreqAlpha: npt.ArrayLike = np.exp(
            -1 / (self.Fss_fft * timeConstants)
        )

    def get_hannWindow(self, NF: int = 1025) -> npt.ArrayLike:
        """
        Returns a scaled Hann window.

        Inputs:
        ----------
        `NF` : int
            number of points of the window

        Returns:
        --------
        `h`:array-like
            Hann window
        """
        n = np.arange(NF)

        h = (
            1 - np.cos((2 * np.pi * n) / (NF - 1))
        ) * 0.5  # Hann window of maximum value 1
        h *= np.sqrt(8 / 3)  # Scaling the output
        return h

    def apply_STFT(self, x: npt.ArrayLike) -> npt.ArrayLike:
        """Computes the STFT of x"""

        numChannels, numSamples = x.shape

        # Number of frames
        numFrames = (numSamples - (self.NF - self.hopSize)) // self.hopSize

        F = 1025

        # Initializing the output array
        X = np.zeros((numChannels, F, numFrames), dtype=np.float32)

        w = np.reshape(self.window, (1, self.NF)) / self.NF

        x_framed = np.zeros((numChannels, self.NF, numFrames))
        start_idx = 0
        for t in range(numFrames):
            end_idx = min(start_idx + self.NF, numSamples)
            x_framed[:, :, t] = x[:, start_idx:end_idx]
            start_idx += self.hopSize

        w = np.reshape(self.window, (1, self.NF, 1)) / self.NF
        X = np.abs(np.fft.rfft(x_framed * w, axis=1))

        return X

    def get_earFilter(self) -> npt.ArrayLike:
        """
        Returns the weights of the spectral weighting filter:

        Returns:
        --------
        `W` : array-like
            array of ear filter weights
        """
        # self.f_hz[0]=1

        mf = np.ones_like(self.f_hz)  # This trick is to avoid dividing by zero
        mf[1:] = self.f_hz[1:] * 0.001

        W_dB = -2.184 * np.power(mf, -0.8)
        W_dB += 6.5 * np.exp(-0.6 * np.square(mf - 3.3))
        W_dB += -0.001 * np.power(mf, 3.6)
        W = self.idB20(W_dB)
        W[0] = 0
        return W.reshape(1, self.numFftBands, 1)

    def hzToBark(self, f_hz: npt.ArrayLike) -> npt.ArrayLike:
        """
        Converts from Hertz to Bark scale.

        Inputs:
        ----------
        `f_hz` : array-like
            Frequencies in Hertz.

        Returns:
        --------
        `z`: array-like, same size as `f`
            frequencies converted to Bark scale.
        """
        z = 7 * np.arcsinh(f_hz / 650)
        return z

    def barkToHz(self, z: npt.ArrayLike) -> npt.ArrayLike:
        """
        Converts frequencies from Bark scale to Hertz.

        Inputs:
        ----------
        `z` : array-like
            Frequencies in Bark scale.

        Returns:
        --------
        `f_hz` : array-like
            FRequencies in Hertz.
        """
        f_hz = 650 * np.sinh(z / 7)
        return f_hz

    def get_BarkBandsFreqs(
        self,
        lowerBound_hz: float = 80,
        upperBound_hz: float = 18000,
    ) -> tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike, int]:
        """
        Returns the lower, higher and center frequencies of all Bark bands.
        """
        # band_idx = np.arange(self.numBarkBands)

        # z_L = self.hzToBark(lowerBound_hz)
        # z_U = self.hzToBark(upperBound_hz)

        # z_l = z_L + band_idx * self.barkWidth
        # z_u = np.minimum(z_l + self.barkWidth, z_U)
        # z_c = (z_u + z_l) / 2

        # f_l = self.barkToHz(z_l).reshape(self.numBarkBands, 1)
        # f_u = self.barkToHz(z_u).reshape(self.numBarkBands, 1)
        # f_c = self.barkToHz(z_c).reshape(self.numBarkBands, 1)

        f_l = critical_bands.f_l.reshape(self.numBarkBands, 1)

        f_c = critical_bands.f_c.reshape(self.numBarkBands, 1)

        f_u = critical_bands.f_u.reshape(self.numBarkBands, 1)

        return f_l, f_c, f_u

    def get_barkBinGrouping(
        self,
    ) -> npt.ArrayLike:

        f_u = self.f_u.reshape(1, 1, self.numBarkBands, 1)
        f_l = self.f_l.reshape(1, 1, self.numBarkBands, 1)

        k = np.arange(self.numFftBands).reshape(1, self.numFftBands, 1, 1)
        i = np.arange(self.numBarkBands).reshape(1, 1, self.numBarkBands, 1)

        U = np.minimum(f_u, (2 * k + 1) * self.sr_hz / self.NF * 0.5)
        U = U - np.maximum(f_l, (2 * k - 1) * self.sr_hz / self.NF * 0.5)
        U = np.maximum(U, 0) * self.NF / self.sr_hz
        # U=U.reshape(1,  self.numFftBands, self.numBarkBands, 1)
        return U

    def get_internalNoise(self, f) -> npt.ArrayLike:
        zeros = np.where(f < 1e-12)
        # other = np.where(f >= 1e-12)
        Ein_dB = np.ones_like(f)
        # Ein_dB[other] = np.power(f[other] * 0.001, -0.8)
        Ein_dB = 1.456 * np.power(f * 0.001, -0.8)
        Ein = self.idB10(Ein_dB)
        # Ein[zeros] = 0
        return Ein

    def frequencySpreading(self, E):
        numChannels, numBarkBands, numFrames = E.shape

        i = np.arange(numBarkBands, dtype=np.int32).reshape(1, numBarkBands, 1, 1)
        l = np.arange(numBarkBands, dtype=np.int32).reshape(1, 1, numBarkBands, 1)

        # Indices array
        i_l = i - l

        # Reshaping previous arrays
        f_c_rs = self.f_c.reshape(1, 1, numBarkBands, 1)
        E = E.reshape(numChannels, 1, numBarkBands, numFrames)

        a_L = np.power(10, 2.7 * self.barkWidth)

        a_U = np.power(10, -2.4 * self.barkWidth)

        a_C = np.power(10, -23 * self.barkWidth / f_c_rs)

        a_E = np.power(E, 0.2 * self.barkWidth)

        S = np.where(i_l <= 0, a_L, a_U * a_C * a_E)
        S = np.power(S, i_l)

        A = np.sum(S, axis=1, keepdims=True)
        S = S / A

        Es = np.power(S * E, 0.4)

        Es = np.sum(Es, axis=2)
        Es = np.power(Es, 2.5) / self.B_s
        return Es

    def AR_filter(self, X, alpha):
        return self.AR_filter_jit(X, alpha)

    # @njit
    @staticmethod
    def AR_filter_jit(X, alpha):
        numChannels, numBands, numFrames = X.shape

        out = np.zeros_like(X)
        out_prev = np.zeros((numChannels, numBands))

        alpha = alpha.reshape(1, numBands)
        for t in range(numFrames):
            out[:, :, t] = alpha * out_prev + (1 - alpha) * X[:, :, t]
            out_prev = out[:, :, t]

        return out

    def timeDomainSpreading(self, Es):
        E_f = self.AR_filter(Es, self.timeToFreqAlpha)
        out_pattern = np.maximum(E_f, Es)
        return out_pattern

    def get_TimeConstants(self, f, tauMin_s=0.008, tau100_s=0.03):
        tau_s = np.zeros((1, self.numBarkBands))

        tau_s = tauMin_s + 100 / f * (tau100_s - tauMin_s)

        return tau_s

    def apply_loudnessScaling(self, x):
        """
        Scales the data so that it corresponds to a 16 bits encoded waveform
        """
        return x * 32768 / self.Amax

    def apply_earFilter(self, X):
        """
        Applies the ear model filter to a STFT"""
        Xw2 = np.square(np.abs(X) * self.W * self.G_L)
        return Xw2

    def apply_internalNoise(self, Eb):
        """
        Applies the internal noise model to an excitation pattern
        """
        E = Eb + self.E_IN.reshape(1, self.numBarkBands, 1)
        return E

    def apply_frequencyGrouping(self, Xw2):
        """Applies the frequency grouping to a squared magnitude weighted STFT"""
        return apply_frequencyGrouping_jit(Xw2, self.U)

    def apply_frequencyGrouping_efficient(self, Xw2):
        numChannels, _, numFrames = Xw2.shape
        numBarkBands = self.U.shape[2]

        Ea = np.zeros((numChannels, numBarkBands, numFrames))

        k_l = 0

        for barkBand_idx in range(numBarkBands):
            while self.U[0, k_l, barkBand_idx, 0] == 0.0:
                k_l += 1
            k_u = k_l
            while self.U[0, k_u, barkBand_idx, 0] > 0.0:
                k_u += 1
            temp = np.sum(
                self.U[:, k_l:k_u, barkBand_idx, :] * Xw2[:, k_l:k_u, :],
                axis=1,
            )
            Ea[:, barkBand_idx, :] = temp

        Emin = 1e-12
        Eb = np.maximum(Ea, Emin)
        return Eb

    def apply_stftToPatterns(self, Xw2):
        Eb = self.apply_frequencyGrouping_efficient(Xw2)

        E = self.apply_internalNoise(Eb)

        Es = self.frequencySpreading(E)

        EsTilde = self.timeDomainSpreading(Es)
        return Es, EsTilde

    def apply_waveformToStft(self, x):
        x = self.apply_loudnessScaling(x)
        X = self.apply_STFT(x)
        return X

    def compute_noisePatterns(self, Xw2_T, Xw2_R):
        Xw2N_squared = Xw2_T - 2 * np.sqrt(Xw2_T * Xw2_R) + Xw2_R
        EbN = self.apply_frequencyGrouping_efficient(Xw2N_squared)
        return EbN


def frequencySpreading_jit(E, barkwidth, f_c, B_s):
    numChannels, numBarkBands, numFrames = E.shape

    i = np.arange(numBarkBands, dtype=np.int32).reshape(1, numBarkBands, 1, 1)
    l = np.arange(numBarkBands, dtype=np.int32).reshape(1, 1, numBarkBands, 1)

    # Indices array
    i_l = i - l

    # Reshaping previous arrays
    f_c_rs = f_c.reshape(1, 1, numBarkBands, 1)
    E = E.reshape(numChannels, 1, numBarkBands, numFrames)

    a_L = np.power(10, 2.7 * barkwidth)

    a_U = np.power(10, -2.4 * barkwidth)

    a_C = np.power(10, -23 * barkwidth / f_c_rs)

    a_E = np.power(E, 0.2 * barkwidth)

    S = np.where(i_l <= 0, a_L, a_U * a_C * a_E)
    S = np.power(S, i_l)

    A = np.sum(S, axis=1, keepdims=True)
    S = S / A

    Es = np.power(S * E, 0.4)

    Es = np.sum(Es, axis=2)
    Es = np.power(Es, 2.5) / B_s
    return Es


@njit
def apply_frequencyGrouping_jit(Xw2: npt.ArrayLike, U: npt.ArrayLike) -> npt.ArrayLike:
    numChannels, _, numFrames = Xw2.shape
    numBarkBands = U.shape[2]

    Ea = np.zeros((numChannels, numBarkBands, numFrames))

    k_l = 0

    for barkBand_idx in range(numBarkBands):
        while U[0, k_l, barkBand_idx, 0] == 0.0:
            k_l += 1
        k_u = k_l
        while U[0, k_u, barkBand_idx, 0] > 0.0:
            k_u += 1
        temp = np.sum(
            U[:, k_l:k_u, barkBand_idx, :] * Xw2[:, k_l:k_u, :],
            axis=1,
        )
        Ea[:, barkBand_idx, :] = temp

    Emin = 1e-12
    Eb = np.maximum(Ea, Emin)
    print(np.where(Eb < 1e-12))
    return Eb
