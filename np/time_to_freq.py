import numpy as np
import numpy.typing as npt

class Mixin:
    def get_hannWindow(self, NF:int=1025) -> npt.ArrayLike:
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

        h= (1-np.cos((2*np.pi*n)/(NF-1)))*.5 #Hann window of maximum value 1
        h = np.sqrt(8/3)*h #Scaling the output
        return h

    def apply_STFT(self, x:npt.ArrayLike)->npt.ArrayLike:
        """Computes the STFT of x"""

        numChannels, numSamples= x.shape


        # Number of frames
        T = (numSamples-(self.NF-self.hopSize))//self.hopSize

        F = 1025
        
        #Initializing the output array
        X = np.zeros((numChannels, F, T)) 

        for t in range(T):
            for channel in range(numChannels):
                x_w = x[channel,t*self.hopSize:t*self.hopSize+self.NF]
                x_w = x_w*self.window
                X[channel, :, t]= np.abs(np.fft.rfft(x_w))
        
        return X/self.NF

    def get_earFilter(self) -> npt.ArrayLike:
        """
        Returns the weights of the spectral weighting filter:
        
        Returns:
        --------
        `W` : array-like
            array of ear filter weights
        """
        #self.f_hz[0]=1

        mf = np.ones_like(self.f_hz) #This trick is to avoid dividing by zero
        mf[1:] = self.f_hz[1:]*.001

        W_dB = -2.184*np.power(mf, -.8)
        W_dB += 6.5*np.exp(-.6*np.square(mf-3.3))
        W_dB += -.001*np.power(mf, 3.6)
        W = self.idB20(W_dB)
        W[0]=0
        return W.reshape(1, self.numFftBands, 1)


    def hzToBark(self,f_hz:npt.ArrayLike) -> npt.ArrayLike:
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
        z = 7*np.arcsinh(f_hz/650)
        return z

    def barkToHz(self, z:npt.ArrayLike) -> npt.ArrayLike:
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
        f_hz = 650*np.sinh(z/7)
        return f_hz


    def get_BarkBandsFreqs(
            self,
            lowerBound_hz:float=80,
            upperBound_hz:float=18000,
        ) -> tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike, int]:
        """
        Returns the lower, higher and center frequencies of all Bark bands.
        """
        band_idx = np.arange(self.numBarkBands)

        z_L = self.hzToBark(lowerBound_hz)
        z_U = self.hzToBark(upperBound_hz)

        z_l = z_L + band_idx*self.barkWidth
        z_u =np.minimum(z_l+self.barkWidth, z_U)
        z_c = (z_u+z_l)/2

        f_l = self.barkToHz(z_l).reshape(self.numBarkBands, 1)
        f_u = self.barkToHz(z_u).reshape(self.numBarkBands, 1)
        f_c = self.barkToHz(z_c).reshape(self.numBarkBands, 1)

        return f_l, f_c, f_u

    def get_barkBinGrouping(
            self,
            )->npt.ArrayLike:

        f_u = self.f_u.reshape(1, 1, self.numBarkBands, 1)
        f_l = self.f_l.reshape(1, 1, self.numBarkBands, 1)
        U = np.zeros((1, self.numBarkBands, self.numFftBands, 1))

        k = np.arange(self.numFftBands).reshape(1, self.numFftBands, 1, 1)
        i = np.arange(self.numBarkBands).reshape(1, 1, self.numBarkBands, 1)

        U = np.minimum(f_u, (2*k+1)*self.sr_hz/self.NF*.5)
        U = U- np.maximum(f_l, (2*k-1)*self.sr_hz/self.NF*.5)
        U = np.maximum(U, 0)*self.NF/self.sr_hz 
        #U=U.reshape(1,  self.numFftBands, self.numBarkBands, 1)
        return U


    def get_internalNoise(self, f)->npt.ArrayLike:
        zeros = np.where(f<1e-12)
        other = np.where(f>=1e-12)
        Ein_dB = np.ones_like(f)
        Ein_dB[other] = np.power(f[other]*.001, -0.8)
        Ein_dB=Ein_dB*1.456
        #Ein_dB = 1.456*np.power(f*.001, -0.8)
        Ein = self.idB10(Ein_dB)
        Ein[zeros]=0
        return Ein


    def frequencySpreading(self, E):
        numChannels = E.shape[0]
        num_frames= E.shape[2]

        i = np.arange(self.numBarkBands, dtype='int32').reshape(1, self.numBarkBands, 1, 1)
        l = np.arange(self.numBarkBands, dtype='int32').reshape(1, 1, self.numBarkBands, 1)
        
        # Indices array
        i_l = i-l

        # Reshaping previous arrays
        f_c_rs = self.f_c.reshape(1, 1, self.numBarkBands, 1)
        E = E.reshape(numChannels, 1, self.numBarkBands, num_frames)

        S_dB = np.zeros((numChannels, self.numBarkBands, self.numBarkBands, num_frames))



        S_dB = -24 -230/f_c_rs+2*np.log10(E)*np.ones_like(S_dB)
        S_dB = S_dB*i_l*self.barkWidth

        idx:npt.ArrayLike = np.asarray(i_l<=0)*np.ones_like(S_dB)
        idx = idx.nonzero()
        S_dB[idx] = ((27*i_l*self.barkWidth)*np.ones((numChannels, self.numBarkBands, self.numBarkBands, num_frames)))[idx]
        
        S = self.idB10(S_dB)

        
        A = np.sum(S, axis=1).reshape(numChannels, 1, self.numBarkBands, num_frames)
        S = S/A

        Es = np.power(S*E.reshape(numChannels, 1, self.numBarkBands, num_frames), 0.4)
        
        Es = np.sum(Es, axis=2)
        Es = np.power(Es, 2.5)/self.B_s
        return Es


    def AR_filter(self, X, alpha):
        numChannels,numBands,numFrames=X.shape

        out = np.zeros_like(X)
        out_prev=np.zeros((numChannels, numBands))

        alpha=alpha.reshape(1, numBands)
        for t in range(numFrames):
            out[:,:,t] = alpha*out_prev +(1-alpha)*X[:,:,t]
            out_prev=out[:,:,t]
        
        return out


    def timeDomainSpreading(self, Es):
        E_f = self.AR_filter(Es, self.timeToFreqAlpha)
        out_pattern=np.maximum(E_f, Es)
        return out_pattern
        

    def frequencySmoothing(self, correlation:npt.ArrayLike):
        numChannels, numBands, numFrames = correlation.shape

        k = np.arange(self.numBarkBands, dtype=np.int16).reshape(1, self.numBarkBands, 1, 1)

        M1 = np.minimum(self.FFTM1, k)
        M2 = np.minimum(self.FFTM2, self.numBarkBands-1-k)

        #indices are (k, i)
        i = np.arange(self.numBarkBands, dtype=np.int16).reshape(1, 1, self.numBarkBands, 1)
        passingMatrix = np.zeros((1, numBands, numBands, 1))
        #idx = np.where()
        passingMatrix[(i>=k-M1)]=1.
        passingMatrix[(i>k+M2)]=0

        out = correlation.reshape(numChannels, numBands, 1, numFrames)*passingMatrix
        norm=1/(M1+M2+1).reshape(1, self.numBarkBands, 1)
        out = np.sum(out, axis=2)/norm #Sum over i and normalize
        
        return out

    def get_thresholdIndex(self, f):
        s_dB = -2-2.05*np.arctan(f*.00025) - .75*np.arctan(np.square(f*0.000625))
        s=self.idB10(s_dB)
        return s

    def get_excitationThreshold(self, f):
        Et_dB = 3.64*np.power((f/1000), -.8)
        Et = self.idB10(Et_dB)
        return Et
    
    def get_TimeConstants(self, f, tauMin_s = .008, tau100_s=.03):
        tau_s = np.zeros((1, self.numBarkBands))

        tau_s = tauMin_s+100/f*(tau100_s-tauMin_s)

        return tau_s

    def apply_loudnessScaling(self, x):
        """
        Scales the data so that it corresponds to a 16 bits encoded waveform
        """
        return x/self.Amax*32768

    def apply_earFilter(self, X):
        """
        Applies the ear model filter to a STFT"""
        Xw2 = np.square(np.abs(X)*self.W*self.G_L)
        return Xw2

    def apply_internalNoise(self, Eb):
        """
        Applies the internal noise model to an excitation pattern
        """
        E=Eb+self.E_IN.reshape(1, self.numBarkBands, 1)
        return E

    def apply_frequencyGrouping(self, Xw2):
        """Applies the frequency grouping to a squared magnitude weighted STFT"""
        numChannels, _, numFrames = Xw2.shape
        Xw2 = Xw2.reshape(numChannels, self.numFftBands, 1, numFrames)
        Ea = Xw2*self.U
        Ea = np.sum(Ea, axis=1)
        Emin = 1e-12*np.ones_like(Ea)
        Eb = np.maximum(Ea, Emin)
        return Eb

    def apply_stftToPatterns(self, Xw2):
        Eb = self.apply_frequencyGrouping(Xw2)

        E = self.apply_internalNoise(Eb)
        
        Es = self.frequencySpreading(E)
        
        EsTilde = self.timeDomainSpreading(Es)
        return Es, EsTilde

    def apply_waveformToStft(self, x):
        x = self.apply_loudnessScaling(x)
        X = self.apply_STFT(x)
        return X

    def compute_noisePatterns(self, Xw2_T, Xw2_R):
        Xw2N_squared = Xw2_T-2*np.sqrt(Xw2_T*Xw2_R)+Xw2_R
        EbN = self.apply_frequencyGrouping(Xw2N_squared)
        return EbN