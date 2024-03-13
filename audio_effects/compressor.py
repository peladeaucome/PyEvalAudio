from .main import AudioEffect
import numpy as np
from numba import njit


def compute_timeConstants(tau_s, samplerate):
    return np.exp(-1 / (tau_s * samplerate))


def dB20(x, eps=1e-6):
    return 20 * np.log10(np.maximum(np.abs(x), eps))


def idB20(x_dB):
    return np.power(10, x_dB / 20)


class Compressor(AudioEffect):
    def __init__(
        self,
        threshold_dB,
        ratio,
        attackTime_ms,
        releaseTime_ms,
        knee_dB,
        samplerate=44100,
    ):
        super().__init__(samplerate)

        self.threshold_dB = threshold_dB
        self.ratio = ratio
        self.attackTime_ms = attackTime_ms
        self.releaseTime_ms = releaseTime_ms
        self.knee_dB = knee_dB

        self.alphaA, self.alphaR = self.get_timeConstants()

    def get_timeConstants(self):
        alphaA = compute_timeConstants(self.attackTime_ms / 1000, self.samplerate)
        alphaR = compute_timeConstants(self.releaseTime_ms / 1000, self.samplerate)
        return alphaA, alphaR

    def compute_dynamicGain(self,x):

        xG = dB20(x, eps=1e-6)

        yG = self.gainComputer(xG)

        xL = xG - yG

        yL = self.levelSmoothing(xL)

        c = idB20(-yL)

        return c
    

    def process(self, x):
        c=self.compute_dynamicGain(x)
        return x * c
    
    def sidechain(self, xMain, xSide):
        c=self.compute_dynamicGain(xSide)
        return xMain * c
    

    def gainComputer(self, xG):
        T = self.threshold_dB
        R = self.ratio
        W = self.knee_dB

        yG = xG.copy()

        idx = np.where(np.abs(xG - T) <= W / 2)
        yG[idx] = xG[idx] + (1 / R - 1) * np.square(xG[idx] - T + W / 2) / (2 * W)

        idx = np.where(xG > (T + W / 2))
        yG[idx] = T + (xG[idx] - T) / R
        return yG

    def levelSmoothing(self, xL):
        return levelSmoothing_jit(xL, self.alphaA, self.alphaR)


@njit
def levelSmoothing_jit(xL, alphaA, alphaR):
    yprev = 0
    yL = np.zeros_like(xL)
    for n in range(len(xL)):
        if xL[n] > yprev:
            yprev = alphaA * yprev + (1 - alphaA) * xL[n]
        else:
            yprev = alphaR * yprev + (1 - alphaR) * xL[n]
        yL[n] = yprev
    return yL
