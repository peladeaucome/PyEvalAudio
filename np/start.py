import numpy as np
import matplotlib.pyplot as plt
import numpy.typing as npt
import MOVs
import pattern_processing
import utils
import time_to_freq
import MOVs

class PEAQ_numpy(
    utils.Mixin,
    time_to_freq.Mixin,
    pattern_processing.Mixin,
    MOVs.Mixin
    ):
    """The advanced-mode is not used for now."""
    def __init__(
            self,
            mode='basic',
            Amax=32768
    ):
        
        self.NF=2048 # Window size
        self.hopSize=1024 # Frame advance
        self.sr_hz = 48000 # Sample rate

        self.mode=mode
        if mode=='basic':
            self.numBarkBands=109
            self.barkWidth=.25
            self.FFTM1, self.FFTM2 = 3, 4
        elif mode=='advanced':
            self.numBarkBands=55
            self.barkWidth=.5
            self.FFTM1, self.FFTM2= (1, 2)
            self.FBM1, self.FBM2 = (1, 1)
        
        # A_max : maximum value that a file can have
        self.Amax:float=Amax

        # FFT frequencies
        self.numFftBands:int=1025
        self.f_hz:npt.ArrayLike = np.fft.rfftfreq(n=2048, d = 1/self.sr_hz)
        
        # Hann window
        self.window=npt.ArrayLike = self.get_hannWindow(NF=self.NF)

        # loudness scaling
        self.G_L:float=3.504
        #self.G_L:float =91.55

        # FFT weighting filter
        self.W:npt.ArrayLike = self.get_earFilter()

        # Critical bands of the FFT model
        self.f_l, self.f_c, self.f_u = self.get_BarkBandsFreqs()

        # Grouping matrix
        self.U=self.get_barkBinGrouping()

        # Internal noise
        self.E_IN=self.get_internalNoise(f=self.f_c).reshape(1,self.numBarkBands, 1)

        # Precompute frequency spreading normalization factor
        self.B_s=1
        self.B_s = self.frequencySpreading(
            E=np.ones((1, self.numBarkBands, 1))
        )#.reshape(1, self.numBarkBands)

        # Precompute time spreading factors
        self.Fss_fft = self.sr_hz*2/self.NF

        timeConstants = self.get_TimeConstants(
            f=self.f_c[:,0],
            tau100_s=0.030,
            tauMin_s=0.008
        ).reshape(1, self.numBarkBands)
        self.timeToFreqAlpha:npt.ArrayLike = np.exp(-1/(self.Fss_fft*timeConstants))

        timeConstants = self.get_TimeConstants(
            f=self.f_c[:,0],
            tau100_s=0.030,
            tauMin_s=0.008
        ).reshape(1, self.numBarkBands)
        self.patternProcessingAlpha:npt.ArrayLike = np.exp(-1/(self.Fss_fft*timeConstants))

        # s
        self.s = self.get_thresholdIndex(f=self.f_c).reshape(1,self.numBarkBands, 1)
        # E_t
        self.Et = self.get_excitationThreshold(f=self.f_c).reshape(1,self.numBarkBands, 1)

        # 5.2.1 Delayed Averaging
        tauDel = 0.5
        self.Ndel = np.ceil(tauDel*self.Fss_fft)

        tauOff = 0.05
        self.Noff = np.ceil(tauOff*self.Fss_fft)

    
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
            Es_T:npt.ArrayLike,
            Es_R:npt.ArrayLike,
        ):
        numChannels, numBands, numFrames = Es_R.shape
        
        EsTilde_T=self.timeDomainSpreading(
            Es_T)
        EsTilde_R=self.timeDomainSpreading(
            Es_R)

        EP_T, EP_R = self.excitationPatternProcessing(
            EsTilde_T,
            EsTilde_R
        )
        
        M_T, M_R, Ebar_R = self.modulationPatternProcessing(
            Es_T,
            Es_R
        )

        Ntot_T, Ntot_R = self.loudnessCalculation(
            EsTilde_T,
            EsTilde_R
        )

        # Returning all the values needed to compute the MOVs
        return EP_T, EP_R, M_T, M_R, Ebar_R, Ntot_T, Ntot_R

    def two_f_model(self, x_T, x_R):
        X_T, X_R, Es_T, Es_R, EsTilde_T, EsTilde_R, EbN = self.timeToFrequencyDomain(x_T, x_R)


        patt = self.patternProcessing(Es_T, Es_R)
        
        
        return patt

    def compute_PEAQ(self, x_T, x_R):
        X_T, X_R, Es_T, Es_R, EsTilde_T, EsTilde_R, EbN = self.timeToFrequencyDomain(x_T, x_R)

        patt = self.patternProcessing(Es_T, Es_R)
        EP_T, EP_R, M_T, M_R, Ebar_R, Ntot_T, Ntot_R=patt

        start_sampleIdx, end_sampleIdx = self.get_dataBoundary(x_T=x_T, x_R=x_R)
        start_frameIdx, end_frameIdx = start_sampleIdx//self.hopSize, end_sampleIdx//self.hopSize
        
        Ntot_T, Ntot_R=crop_multiple(Ntot_T, Ntot_R, start_idx=start_frameIdx, end_idx=end_frameIdx)

        return patt



class Data():
    def __init__(self, x, Amax):
        pass



def crop_multiple(*X_list, start_idx, end_idx):
    out=[]
    for X in X_list:
        out.append(X[...,start_idx: end_idx])
    return tuple(out)



if __name__ == '__main__':
    frameSize = 2048
    Amax=32768
    peaq = PEAQ_numpy(mode='basic', Amax=Amax)

    cm = 1/2.54
    figsize = (12*cm, 8*cm)

    plt.figure(figsize=figsize)
    plt.title('Fig. 1: Outer and middle ear response.')
    plt.plot(peaq.f_hz, peaq.dB20(peaq.W[0,:,0]), 'b')
    plt.xlim(0, 12000)
    plt.ylim(-20, 10)
    plt.grid()
    plt.xlabel('Response (dB)')
    plt.ylabel('Frequency (Hz)')
    plt.tight_layout()
    plt.savefig('Figures/numpy/fig1.pdf', format='pdf')
    plt.close()

    plt.figure(figsize=figsize)
    plt.title("Fig. 2: Internal noise contribution")
    plt.plot(peaq.f_c[:,0],peaq.dB10(peaq.E_IN[0, :, 0]) , '.-b')
    plt.xlim(0, 12000)
    plt.ylim(0, 20)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (dB)')
    plt.grid()
    plt.tight_layout()
    plt.savefig('Figures/numpy/fig2.pdf', format='pdf')
    plt.close()

    plt.figure(figsize=figsize)
    plt.title("Fig. 3: Normalization factor for the Basic version of PEAQ")
    plt.plot(peaq.f_l,peaq.dB10(peaq.B_s[0]) , '.-b')
    plt.xlim(0, 18000)
    plt.ylim(0, 14)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (dB)')
    plt.yticks([0, 2, 4, 6, 8, 10, 12, 14])
    plt.xticks([0, 5000, 10000, 15000])
    plt.grid()
    plt.tight_layout()
    plt.savefig('Figures/numpy/fig3.pdf', format='pdf')
    plt.close()

    
    plt.figure(figsize=figsize)
    for i in range(0,109,4):
        E = np.zeros((1, peaq.numBarkBands, 1))+1e-6
        E[0,i,:]=1000000
        out=peaq.frequencySpreading(E=E)
        out=out*peaq.B_s

        plt.plot(peaq.f_l, peaq.dB10(out[0]), linewidth=1)
    plt.grid()
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (dB)')
    plt.title('Fig. 4: Spreading functions (60dB signal level)')
    plt.yticks(np.arange(13)*5)
    plt.xlim(0, 8000)
    plt.ylim(0, 60)
    plt.tight_layout()
    plt.savefig('Figures/numpy/fig4.pdf', format='pdf')
    plt.close()

    plt.figure(figsize=figsize)
    f = np.linspace(1, 12000, 1000)
    tau = peaq.get_TimeConstants(f)
    plt.plot(f, tau*1000, 'b')
    bands=peaq.f_l
    plt.plot(bands, peaq.get_TimeConstants(bands)*1000, 'xb')
    plt.xlim(0, 12000)
    plt.ylim(0, 50)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Time constant (ms)')
    plt.title('Fig. 5: Time constants as a function of frequency.')
    plt.tight_layout()
    plt.grid()
    #plt.show()
    plt.savefig('Figures/numpy/fig5.pdf', format='pdf')
    plt.close()

    bands=peaq.f_c
    plt.figure(figsize=figsize)
    plt.title('Fig. 9 : Excitation threshold and threshold index.')
    plt.plot(f, peaq.dB10(peaq.get_excitationThreshold(f)), 'b')
    plt.plot(bands, peaq.dB10(peaq.get_excitationThreshold(bands)), 'bx')
    plt.text(x=6000, y=2.5, s='Excitation threshold')
    plt.plot(f, peaq.dB10(peaq.get_thresholdIndex(f)), 'g')
    plt.plot(bands, peaq.dB10(peaq.get_thresholdIndex(bands)), 'gx')
    plt.text(x=6000, y=-3.5, s='Threshold index')
    plt.xlim(0, 12000)
    plt.ylim(-10, 20)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Response (dB)')
    plt.grid()
    plt.tight_layout()
    plt.savefig('Figures/numpy/fig9.pdf', format='pdf')
    #plt.show()
    plt.close()
    
    with open('Center frequencies.txt', 'w') as f:
        for freq in peaq.f_c[:,0]:
            f.write(str(round(freq, 3)))
            f.write('\n')
    
    f = 1000
    numSamples=peaq.NF*25
    n = np.arange(numSamples)
    n = n.reshape(1, numSamples)
    amp=Amax*100/peaq.idB20(92)
    #print(amp)
    x_T = np.sin(n*2*np.pi*f/peaq.sr_hz)*amp # test signal
    x_R = np.sin(n*2*np.pi*f/peaq.sr_hz)*amp*np.sqrt(10) # ref signal
    #x=np.zeros((2, 2048))
    
    
    EP_T, EP_R, M_T, M_R, Ebar_R, Ntot_T, Ntot_R = peaq.compute_PEAQ(x_T, x_R)
    
    plt.figure(figsize=figsize)
    plt.plot(peaq.f_l, peaq.get_maskThreshold(), 'b-x')
    plt.title('Fig. 12 : Masking offset as a function of frequency')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Masking offset (dB)')
    plt.grid()
    plt.tight_layout
    plt.xlim(0, 12000)
    plt.ylim(0, 8)
    plt.savefig('Figures/numpy/fig12.pdf', format='pdf')
    #plt.show()
    plt.close()
    

    print(f'Loudness of a 1kHz sine wave of loudness 92dB SPL: {round(np.mean(Ntot_T),3)} sones')
    print(f'Loudness of a 1kHz sine wave 10 dB louder:         {round(np.mean(Ntot_R),3)} sones')
    print(f'Loudness difference:                               {round(np.mean(Ntot_R)-np.mean(Ntot_T), 3)} sones')
    #print(Ntot_R)
    #print(out)

#

