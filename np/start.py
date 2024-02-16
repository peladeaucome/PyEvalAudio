import numpy as np
import matplotlib.pyplot as plt
import numpy.typing as npt
import MOVs
import pattern_processing
import utils
import time_to_freq
import MOVs
import ODG

class PEAQ(
    utils.Mixin,
    time_to_freq.Mixin,
    pattern_processing.Mixin,
    MOVs.Mixin,
    ODG.Mixin
):
    """The advanced-mode is not used for now."""

    def __init__(self, mode="basic", Amax=32768):

        time_to_freq.Mixin.__init__(self, mode=mode, Amax=Amax)
        pattern_processing.Mixin.__init__(self)
        MOVs.Mixin.__init__(self)
        ODG.Mixin.__init__(self)

    def timeToFrequencyDomain(self, x_T, x_R):
        """
        Does the whole time to frequency domain processing, returning STFTs,
        unsmeared excitation patterns, excitation patterns and noise patterns of
        signals ``x_T`` and ``x_R``.

        Inputs:
        -------
        ``x_T``: Array-like
            Tested signal waveform.
        ``x_R``: Array-like
            Reference signal waveform.

        Outputs:
        ``X_T``: Array-like
            Tested signal STFT
        ``X_R``: Array-like
            Reference signal STFT
        ``Es_T``: Array-like
            Tested signal unsmeared excitation pattern
        ``Es_R``: Array-like
            Reference signal unsmeared excitation pattern
        ``EsTilde_T``: Array-like
            Tested signal excitation pattern
        ``EsTilde_R``: Array-like
            Reference signal excitation pattern
        ``EbN``: Array-like
            Noise excitation pattern
        """

        X_T = self.apply_waveformToStft(x_T)
        X_R = self.apply_waveformToStft(x_R)

        Xw2_T = self.apply_earFilter(X_T)
        Xw2_R = self.apply_earFilter(X_R)

        Es_T, EsTilde_T = self.apply_stftToPatterns(Xw2_T)
        Es_R, EsTilde_R = self.apply_stftToPatterns(Xw2_R)

        EbN = self.compute_noisePatterns(Xw2_T, Xw2_R)

        return X_T, X_R, Es_T, Es_R, EsTilde_T, EsTilde_R, EbN

    def patternProcessing(
        self,
        Es_T: npt.ArrayLike,
        Es_R: npt.ArrayLike,
    ):
        numChannels, numBands, numFrames = Es_R.shape

        EsTilde_T = self.timeDomainSpreading(Es_T)
        EsTilde_R = self.timeDomainSpreading(Es_R)

        EP_T, EP_R = self.excitationPatternProcessing(EsTilde_T, EsTilde_R)

        M_T, M_R, Ebar_R = self.modulationPatternProcessing(Es_T, Es_R)

        Ntot_T, Ntot_R = self.loudnessCalculation(EsTilde_T, EsTilde_R)

        # Returning all the values needed to compute the MOVs
        return EP_T, EP_R, M_T, M_R, Ebar_R, Ntot_T, Ntot_R

    def two_f_model(self, x_T, x_R):
        X_T, X_R, Es_T, Es_R, EsTilde_T, EsTilde_R, EbN = self.timeToFrequencyDomain(
            x_T, x_R
        )

        patt = self.patternProcessing(Es_T, Es_R)

        return patt

    def compute_PEAQ(self, x_T, x_R):
        X_T, X_R, Es_T, Es_R, EsTilde_T, EsTilde_R, EbN = self.timeToFrequencyDomain(
            x_T, x_R
        )

        patt = self.patternProcessing(Es_T, Es_R)
        EP_T, EP_R, M_T, M_R, Ebar_R, Ntot_T, Ntot_R = patt

        startFrame_idx, endFrame_idx = self.get_dataBoundary(x_T=x_T, x_R=x_R)

        Ntot_T, Ntot_R = crop_multiple(
            Ntot_T, Ntot_R, start_idx=startFrame_idx, end_idx=endFrame_idx
        )

        return patt


class Data:
    def __init__(self, x, Amax):
        pass


def crop_multiple(*X_list, start_idx, end_idx):
    out = []
    for X in X_list:
        out.append(X[..., start_idx:end_idx])
    return tuple(out)


if __name__ == "__main__":
    frameSize = 2048
    Amax = 32768
    peaq = PEAQ(mode="basic", Amax=Amax)

    cm = 1 / 2.54
    figsize = (12 * cm, 8 * cm)

    plt.figure(figsize=figsize)
    plt.title("Fig. 1: Outer and middle ear response.")
    plt.plot(peaq.f_hz, peaq.dB20(peaq.W[0, :, 0]), "b")
    plt.xlim(0, 12000)
    plt.ylim(-20, 10)
    plt.grid()
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Response (dB)")
    plt.tight_layout()
    plt.savefig("Figures/numpy/fig1.pdf", format="pdf")
    plt.close()

    plt.figure(figsize=figsize)
    plt.title("Fig. 2: Internal noise contribution")
    plt.plot(peaq.f_c[:, 0], peaq.dB10(peaq.E_IN[0, :, 0]), ".-b")
    plt.xlim(0, 12000)
    plt.ylim(0, 20)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude (dB)")
    plt.grid()
    plt.tight_layout()
    plt.savefig("Figures/numpy/fig2.pdf", format="pdf")
    plt.close()

    plt.figure(figsize=figsize)
    plt.title("Fig. 3: Normalization factor for the Basic version of PEAQ")
    plt.plot(peaq.f_l, peaq.dB10(peaq.B_s[0]), ".-b")
    plt.xlim(0, 18000)
    plt.ylim(0, 14)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude (dB)")
    plt.yticks([0, 2, 4, 6, 8, 10, 12, 14])
    plt.xticks([0, 5000, 10000, 15000])
    plt.grid()
    plt.tight_layout()
    plt.savefig("Figures/numpy/fig3.pdf", format="pdf")
    plt.close()

    plt.figure(figsize=figsize)
    for i in range(0, 109, 4):
        E = np.zeros((1, peaq.numBarkBands, 1)) + 1e-6
        E[0, i, :] = 1000000
        out = peaq.frequencySpreading(E=E)
        out = out * peaq.B_s

        plt.plot(peaq.f_l, peaq.dB10(out[0]), linewidth=1)
    plt.grid()
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude (dB)")
    plt.title("Fig. 4: Spreading functions (60dB signal level)")
    plt.yticks(np.arange(13) * 5)
    plt.xlim(0, 8000)
    plt.ylim(0, 60)
    plt.tight_layout()
    plt.savefig("Figures/numpy/fig4.pdf", format="pdf")
    plt.close()

    plt.figure(figsize=figsize)
    f = np.linspace(1, 12000, 1000)
    tau = peaq.get_TimeConstants(f)
    plt.plot(f, tau * 1000, "b")
    bands = peaq.f_l
    plt.plot(bands, peaq.get_TimeConstants(bands) * 1000, "xb")
    plt.xlim(0, 12000)
    plt.ylim(0, 50)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Time constant (ms)")
    plt.title("Fig. 5: Time constants as a function of frequency.")
    plt.tight_layout()
    plt.grid()
    # plt.show()
    plt.savefig("Figures/numpy/fig5.pdf", format="pdf")
    plt.close()

    bands = peaq.f_c
    plt.figure(figsize=figsize)
    plt.title("Fig. 9 : Excitation threshold and threshold index.")
    plt.plot(f, peaq.dB10(peaq.get_excitationThreshold(f)), "b")
    plt.plot(bands, peaq.dB10(peaq.get_excitationThreshold(bands)), "bx")
    plt.text(x=6000, y=2.5, s="Excitation threshold")
    plt.plot(f, peaq.dB10(peaq.get_thresholdIndex(f)), "g")
    plt.plot(bands, peaq.dB10(peaq.get_thresholdIndex(bands)), "gx")
    plt.text(x=6000, y=-3.5, s="Threshold index")
    plt.xlim(0, 12000)
    plt.ylim(-10, 20)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Response (dB)")
    plt.grid()
    plt.tight_layout()
    plt.savefig("Figures/numpy/fig9.pdf", format="pdf")
    # plt.show()
    plt.close()

    with open("Center frequencies.txt", "w") as f:
        for freq in peaq.f_c[:, 0]:
            f.write(str(round(freq, 3)))
            f.write("\n")

    plt.figure(figsize=figsize)
    plt.plot(peaq.f_l, peaq.get_maskThreshold(), "b-x")
    plt.title("Fig. 12 : Masking offset as a function of frequency")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Masking offset (dB)")
    plt.grid()
    plt.tight_layout
    plt.xlim(0, 12000)
    plt.ylim(0, 8)
    plt.savefig("Figures/numpy/fig12.pdf", format="pdf")
    # plt.show()
    plt.close()

    f = 1000
    numSamples = peaq.NF * 25
    n = np.arange(numSamples)
    n = n.reshape(1, numSamples)
    amp = Amax * 100 / peaq.idB20(92)
    # print(amp)
    x_T = np.sin(n * 2 * np.pi * f / peaq.sr_hz) * amp  # test signal
    x_R = np.sin(n * 2 * np.pi * f / peaq.sr_hz) * amp * np.sqrt(10)  # ref signal
    # x=np.zeros((2, 2048))

    X_T, X_R, Es_T, Es_R, EsTilde_T, EsTilde_R, EbN = peaq.timeToFrequencyDomain(
        x_T, x_R
    )

    patt = peaq.patternProcessing(Es_T, Es_R)
    EP_T, EP_R, M_T, M_R, Ebar_R, Ntot_T, Ntot_R = patt

    print(
        f"Loudness of a 1kHz sine wave of loudness 92dB SPL: {round(np.mean(Ntot_T),3)} sones"
    )
    print(
        f"Loudness of a 1kHz sine wave 10 dB louder:         {round(np.mean(Ntot_R),3)} sones"
    )
    print(
        f"Loudness difference:                               {round(np.mean(Ntot_R)-np.mean(Ntot_T), 3)} sones"
    )
    # print(Ntot_R)
    # print(out)

#
