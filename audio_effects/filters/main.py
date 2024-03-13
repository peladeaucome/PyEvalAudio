import numpy as np
import scipy.signal as sig
from ..main import *

class Filter(AudioEffect):
    def __init__(self, samplerate):
        super().__init__(samplerate)
        self.b, self.a = self.compute_coeffs()
    
    def compute_coeffs(self):
        return np.array([1]), np.array([1])
    
    def process(self, x):
        return sig.lfilter(b=self.b, a=self.a, x=x)

    def rfft(self, n):
        return np.fft.rfft(a=self.b, n=n)/np.fft.rfft(a=self.a, n=n)
    
    def impulse(self, n):
        y = np.zeros(n)
        y[0] = 1
        return self(y)

class FilterSeries(EffectSeries, Filter):
    def __init__(self, *filters_args):
        super().__init__(self, *filters_args)
    
    def process(self, x):
        return EffectSeries.process(self, x)

    def rfft(self,n):
        out = np.ones(n, dtype=np.complex128)
        for filter in self.effects_list:
            out=out*filter.rfft(filter)

    def check(self, effect):
        super().check(effect)
        if not isinstance(filter, Filter):
            raise TypeError("All effects should be filters")

if __name__=='__main__':
    filt = Filter()

    print(type(filt))