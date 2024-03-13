import numpy as np
from dataclasses import dataclass
import numpy.typing as npt


def get_taper(length_sp):
    n = np.arange(length_sp)

    out = np.square(np.cos(n * np.pi * 0.5 / length_sp))
    return out


def get_P_and_Q(in_sr, out_sr):
    up = out_sr > in_sr

    G = max(in_sr, out_sr)
    L = min(in_sr, out_sr)

    if L == 32000:
        if G == 44100:
            p, q = 441, 320
        elif G == 48000:
            p, q = 3, 2
        elif G == 96000:
            p, q = 3, 1
        elif G == 192000:
            p, q = 6, 1
    elif L == 44100:
        if G == 48000:
            p, q = 160, 147
        elif G == 96000:
            p, q = 320, 147
        elif G == 192000:
            p, q = 640, 147
    elif L == 48000:
        if G == 96000:
            p, q = 2, 1
        elif G == 192000:
            p, q = 4, 1
    elif L == 96000:
        if G == 192000:
            p, q = 2, 1

    if up:
        return p, q
    else:
        return q, p


@dataclass
class Resampler:
    in_samplerate: int
    out_samplerate: int
    taper_ratio: int = 0.05

    def __post_init__(self):
        accepted_samplerates = [32000, 44100, 48000, 96000, 192000]
        if (
            self.in_samplerate not in accepted_samplerates
            or self.out_samplerate not in accepted_samplerates
        ):
            raise NotImplementedError("Please select a valid samplerate")
        if self.in_samplerate == self.out_samplerate:
            raise ValueError("Input samplerate is equal to output samplerate")
        else:
            self.P, self.Q = get_P_and_Q(self.in_samplerate, self.out_samplerate)

    def resample(self, x):
        numChannels, Nin = x.shape

        M = int(2 * np.ceil((Nin+self.in_samplerate*.5) / (self.Q * 2)))

        Nx = M * self.Q
        Ny = M * self.P

        nbins_in = Nx // 2 + 1
        nbins_out = Ny // 2 + 1

        x_fft = np.fft.rfft(x, axis=1, n=Nx)
        ## TAPERING
        if self.out_samplerate > self.in_samplerate:
            Ntaper = int(self.taper_ratio * nbins_in)
            taper = get_taper(Ntaper).reshape(1, Ntaper)
            x_fft[:, -Ntaper:] *= taper
            ## Zero-padding
            y_fft = np.zeros((numChannels, nbins_out), dtype=np.complex128)
            y_fft[:, :nbins_in] = x_fft
        else:
            y_fft = np.zeros((numChannels, nbins_out), dtype=np.complex128)
            y_fft[:, :nbins_out] = x_fft[:, :nbins_out]
            y_fft[:, -1] = np.real(y_fft[:, -1])

            Ntaper = int(self.taper_ratio * nbins_out)
            taper = get_taper(Ntaper).reshape(1, Ntaper)
            y_fft[:, -Ntaper:] *= taper

        # IFFT
        y = np.fft.irfft(y_fft, axis=1, n=Ny)

        # Scaling
        y = y * self.P / self.Q

        # Remove additional zeros
        Nout = int(Nin * self.P / self.Q)
        y = y[:, :Nout]
        return y

    def __call__(self, x):
        return self.resample(x)


if __name__ == "__main__":

    def dB20(x):
        return 20 * np.log(np.maximum(np.abs(x), 1e-10))
    
    def dB10(x):
        return 10 * np.log(np.maximum(np.abs(x), 1e-10))

    @dataclass
    class AudioSignal:
        signal:npt.ArrayLike
        samplerate:int

        def get_time(self):
            return np.arange(self.signal.shape[1])/self.samplerate
        
        def get_rfft(self, n=None):
            out = np.fft.rfft(self.signal, n=n)
            return out
        
        def get_rfftfreq(self, n=None):
            if n is None:
                n=self.signal.shape[1]
            out = np.fft.rfftfreq(n=n, d=1/self.samplerate)
            return out
        
    time_s=1
    x_sr = 44100
    y_sr = 96000
    N = int(x_sr * time_s) + 1

    x = np.zeros((1, N))
    x[:, N // 2 + 1] = 1

    x = AudioSignal(signal=x, samplerate=x_sr)

    RS = Resampler(in_samplerate=x_sr, out_samplerate=y_sr, taper_ratio=0.1)

    y = AudioSignal(signal=RS(x.signal), samplerate=y_sr)

    import matplotlib.pyplot as plt

    
    plt.plot(x.get_time(), dB10(x.signal[0]))
    plt.plot(y.get_time(), dB10(y.signal[0]))
    plt.xlim(time_s/2-.005, time_s/2+.005)
    plt.ylim(-100, 0)
    plt.xlabel("Time (ms)")
    plt.ylabel("Magnitude (dB)")
    plt.show()

    #plt.plot(x.get_rfftfreq(), dB20(x.get_rfft()[0]))
    #plt.plot(y.get_rfftfreq(), dB20(y.get_rfft()[0]))
    #plt.show()
