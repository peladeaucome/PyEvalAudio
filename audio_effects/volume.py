from .main import AudioEffect
import numpy as np
from dataclasses import dataclass

def dB20(x, eps=1e-10):
    return 20*np.log(np.abs(np.maximum(x, eps)))

def idB20(x_dB):
    return np.power(10, x_dB/20)

@dataclass
class Volume(AudioEffect):
    g_dB:float=0

    def process(self, x):
        return x*idB20(self.g_dB)