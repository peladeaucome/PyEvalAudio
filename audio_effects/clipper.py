import numpy as np
from .main import AudioEffect
from dataclasses import dataclass


def dB20(x, eps=1e-6):
    return 20 * np.log10(np.maximum(np.abs(x), eps))


def idB20(x_dB):
    return np.power(10, x_dB / 20)

class SoftClipper(AudioEffect):
    def __init__(self, threshold_dB, knee_dB, samplerate=44100):
        super().__init__(samplerate=samplerate)
        self.threshold_dB=threshold_dB
        self.knee_dB=knee_dB

    def process(self, x):
        xG = dB20(x, eps=1e-6)

        yG = self.gainComputer(xG)

        xL = xG - yG

        c = idB20(-xL)
        return c*x
    
    def gainComputer(self, xG):
        T = self.threshold_dB
        W = self.knee_dB

        yG = xG.copy()

        idx = np.where(np.abs(xG - T) <= W / 2)
        yG[idx] = xG[idx] -np.square(xG[idx] - T + W / 2) / (2 * W)

        idx = np.where(xG > (T + W / 2))
        yG[idx] = T
        return yG
