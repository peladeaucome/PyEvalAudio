import numpy as np
from .main import Filter

class EQBand(Filter):
    def __init__(self, f0, g_dB, Q,  samplerate):
        self.f0 = f0
        self.g_dB = g_dB
        self.Q = Q
        super().__init__(samplerate)

    def compute_intermediateValues(self):
        A = np.power(self.g_dB/40)
        w0 = 2*np.pi*self.f0/self.samplerate
        alpha = np.sin(w0)/(2*self.Q)
        return A, w0, alpha

class Peak(EQBand):
    def compute_coeffs(self):
        A, w0, alpha = self.compute_intermediateValues()
        b = np.zeros(3)
        a = np.zeros(3)
        b[0] =   1 + alpha*A
        b[1] =  -2*np.cos(w0)
        b[2] =   1 - alpha*A
        a[0] =   1 + alpha/A
        a[1] =  -2*np.cos(w0)
        a[2] =   1 - alpha/A
        return b, a

    

class LowShelf(EQBand):
    def compute_coeffs(self):
        A, w0, alpha = self.compute_intermediateValues()
        b = np.zeros(3)
        a = np.zeros(3)
        b[0] =    A*( (A+1) - (A-1)*np.cos(w0) + 2*np.sqrt(A)*alpha )
        b[1] =  2*A*( (A-1) - (A+1)*np.cos(w0)                   )
        b[2] =    A*( (A+1) - (A-1)*np.cos(w0) - 2*np.sqrt(A)*alpha )
        a[0] =        (A+1) + (A-1)*np.cos(w0) + 2*np.sqrt(A)*alpha
        a[1] =   -2*( (A-1) + (A+1)*np.cos(w0)                   )
        a[2] =        (A+1) + (A-1)*np.cos(w0) - 2*np.sqrt(A)*alpha
        return b, a
    

class HighShelf(EQBand):
    def compute_coeffs(self):
        A, w0, alpha = self.compute_intermediateValues()
        b = np.zeros(3)
        a = np.zeros(3)
        b[0] =    A*( (A+1) + (A-1)*np.cos(w0) + 2*np.sqrt(A)*alpha )
        b[1] = -2*A*( (A-1) + (A+1)*np.cos(w0)                   )
        b[2] =    A*( (A+1) + (A-1)*np.cos(w0) - 2*np.sqrt(A)*alpha )
        a[0] =        (A+1) - (A-1)*np.cos(w0) + 2*np.sqrt(A)*alpha
        a[1] =    2*( (A-1) - (A+1)*np.cos(w0)                   )
        a[2] =        (A+1) - (A-1)*np.cos(w0) - 2*np.sqrt(A)*alpha
        return b, a


class EQFilter(Filter):
    def __init__(self, f0, Q,  samplerate):
        self.f0 = f0
        self.Q = Q
        super().__init__(samplerate)

    def compute_intermediateValues(self):
        w0 = 2*np.pi*self.f0/self.samplerate
        alpha = np.sin(w0)/(2*self.Q)
        return w0, alpha

class LowPass(EQFilter):
    def compute_coeffs(self):
        w0, alpha = self.compute_intermediateValues()
        b = np.zeros(3)
        a = np.zeros(3)
        b[0] =  (1 - np.cos(w0))/2
        b[1] =   1 - np.cos(w0)
        b[2] =  (1 - np.cos(w0))/2
        a[0] =   1 + alpha
        a[1] =  -2*np.cos(w0)
        a[2] =   1 - alpha
        return b, a

class HighPass(EQFilter):
    def compute_coeffs(self):
        w0, alpha = self.compute_intermediateValues()
        b = np.zeros(3)
        a = np.zeros(3)
        b[0] =  (1 + np.cos(w0))/2
        b[1] = -(1 + np.cos(w0))
        b[2] =  (1 + np.cos(w0))/2
        a[0] =   1 + alpha
        a[1] =  -2*np.cos(w0)
        a[2] =   1 - alpha
        return b, a