import numpy as np
from .main import AudioEffect



class LowPass(AudioEffect):
    def __init__(self, fstart=10100, fend=10900, eps=1e-3, samplerate=44100):
        super().__init__(samplerate=samplerate)

        self.fstart=fstart
        self.fend=fend
        self.eps=eps
    
    def process(self,x):
        N = len(x)

        # Padding to minimize time aliasing
        N_pad = N+self.samplerate
        K=N_pad//2+1

        kl = int(np.floor(2*K*self.fstart/self.samplerate))
        ku = int(np.ceil(2*K*self.fend/self.samplerate))
        
        X = np.fft.rfft(x, n=N_pad)

        n = np.arange(ku-kl)+kl
        X[ku:] = 0
        X[kl:ku] *= (1-self.eps)*np.square(np.cos((n-kl)/(ku-kl)*np.pi*.5))
        X[kl:]+=self.eps

        y = np.fft.irfft(X, n=N_pad)

        y = y[:N]
        return y
