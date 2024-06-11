import numpy as np
import numpy.typing as npt
import scipy.signal

from numba import njit


class Mixin:
    def __init__(self):
        # 5.2.1 Delayed Averaging
        tauDel = 0.5
        self.N500ms = int(np.ceil(tauDel * self.Fss_fft))

        tauOff = 0.05
        self.N50ms = int(np.ceil(tauOff * self.Fss_fft))

        m_dB = self.get_maskThreshold().reshape(1, self.numBarkBands, 1)
        self.gm = 1 / (self.idB10(m_dB))

    def get_dataBoundary(self, x_T, x_R) -> tuple[int, int]:
        """Finds and returns the data boundaries in frame index"""
        Athr = 200 * (self.Amax / 32768)

        L = 5  # Number of samples in a window
        # Athr_over_L=Athr

        start_idx: int = 0
        val_R = np.sum(np.abs(x_R[:, start_idx : start_idx + L]), axis=1)
        # Finding the starting point
        while np.min(val_R) < Athr:
            val_R += np.abs(x_R[:, start_idx + L]) - np.abs(x_R[:, start_idx])
            start_idx += 1
            # val_R = np.sum(np.abs(x_R[:, start_idx : start_idx + L]), axis=1)

        end_idx: int = x_T.shape[1]
        val_R = np.sum(np.abs(x_R[:, end_idx - L : end_idx]), axis=1)
        while np.max(val_R) < Athr:
            val_R += np.abs(x_R[:, end_idx - L - 1]) - np.abs(x_R[:, end_idx - 1])
            end_idx -= 1
            # val_R = np.sum(np.abs(x_R[:, end_idx - L : end_idx]), axis=1)

        end_idx -= 1

        endMin = self.NF // 2
        startFrame_idx, endFrame_idx = (
            start_idx // self.hopSize,
            (end_idx + 1 - endMin) // self.hopSize,
        )

        if self.verbose:
            print(f"PEAQ data boundaries: {start_idx} - {end_idx}")
        # endFrame_idx += 1
        return startFrame_idx, endFrame_idx

    def compute_modulationChanges(
        self, M_T, M_R, Ebar_R, startFrame_idx=0
    ) -> tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]:

        Ndel = max(0, self.N500ms-startFrame_idx)

        # Delayed averaging
        M_T = M_T[:, :, Ndel :]
        M_R = M_R[:, :, Ndel :]
        Ebar_R = Ebar_R[:, :, Ndel :]

        numChannels, _, N = M_T.shape
        ## WinModDiff1B
        # Instantaneous modulation difference, Eq. (73)
        Mdiff1B = np.abs(M_T - M_R) / (1 + M_R)

        # Scale average over bands, Eq. (74)
        Mdiff1bTilde = 100 * np.mean(Mdiff1B, axis=1)

        # Final MOV, Eq. (75)
        L = 4
        Mdiff1bTilde_sqrt = np.sqrt(Mdiff1bTilde)
        intermediate_term = np.zeros((numChannels, N - L + 1))
        for n in range(L, N + 1):
            intermediate_term[:, n - L] = np.sum(
                Mdiff1bTilde_sqrt[:, n - L : n], axis=1
            )

        intermediate_term = np.power(intermediate_term / L, 4)

        MWdiff1B = np.sqrt(np.sum(intermediate_term, axis=1) / (N - L + 1))
        # Mean across stereo channels
        MWdiff1B = np.mean(MWdiff1B)

        ## AvgModDiff1B
        # Temporal weighting, Eq (78)
        W1B = np.sum(
            Ebar_R / (Ebar_R + 100 * np.power(self.E_IN, 0.3)), axis=1, keepdims=True
        )

        Mdiff1bTilde = Mdiff1bTilde.reshape(numChannels, 1, N)
        # Temporally weighted time average, Eq. (77)
        MAdiff1B = np.sum(Mdiff1bTilde * W1B, axis=2) / np.sum(W1B, axis=2)
        MAdiff1B = MAdiff1B.reshape(numChannels)

        # Mean across channels
        MAdiff1B = np.mean(MAdiff1B)

        ## AvgModDiff2B
        # Instantaneous modulation difference, Eq. (79)
        # Mdiff2B = (M_T - M_R) / (0.01 + M_R)
        # Mdiff2B = np.where(M_T >= M_R, Mdiff2B, -0.1 * Mdiff2B)
        Mdiff2B = np.where(
            M_T >= M_R, (M_T - M_R) / (0.01 + M_R), 0.1 * (M_R - M_T) / (0.01 + M_R)
        )

        # Average over bands, Eq. (80)
        Mdiff2BTilde = 100 * np.mean(Mdiff2B, axis=1, keepdims=True)

        # Temporal weighting, Eq.(82) & (78)
        W2B = np.sum(
            Ebar_R / (Ebar_R + 100 * np.power(self.E_IN, 0.3)),
            axis=1,
            keepdims=True,
        )

        # Temporally weighted time average, Eq. (81)
        MAdiff2B = np.sum(W2B * Mdiff2BTilde, axis=2) / np.sum(W2B, axis=2)
        MAdiff2B = MAdiff2B.reshape(numChannels)

        # Mean across channels
        MAdiff2B = np.mean(MAdiff2B)

        return MWdiff1B, MAdiff1B, MAdiff2B, Ndel

    def compute_partialNoiseLoudness(
        self, EP_T, EP_R, M_R, M_T, alpha, T0, S0
    ) -> npt.ArrayLike:

        beta = np.exp(-alpha * ((EP_T - EP_R) / EP_R))

        E_t = self.E_IN.copy()
        E_t = E_t.reshape(1, self.numBarkBands, 1)

        # Threshold factors, Eq. (88)
        s_R = T0 * M_R + S0
        s_T = T0 * M_T + S0

        # Partial Loudness, Eq. (87)
        a = np.maximum(s_T * EP_T - s_R * EP_R, 0)
        b = E_t + beta * s_R * EP_R

        NL = np.power(E_t / s_T, 0.23) * (np.power(1 + a / b, 0.23) - 1)

        return NL

    def compute_RMSNoiseLoud(self, EP_T, EP_R, M_R, M_T, Ntot_T, Ntot_R, Ndel=0):
        NL = self.compute_partialNoiseLoudness(
            EP_T, EP_R, M_R, M_T, alpha=1.5, T0=0.15, S0=0.5
        )

        # Removing the start according to the loudness test
        loudnessTest_idx = self.loudnessTest(Ntot_T, Ntot_R, Ndel)
        NL = NL[:, :, loudnessTest_idx:]

        # Spectral averaging, Eq. (90)
        NiLTilde = 24 * np.mean(NL, axis=1)

        # Setting to zero if it is less than zero, Eq. (91)
        NTilde = np.maximum(NiLTilde, 0)

        # Computing the MOV, Eq. (92)
        NLrmsB = np.sqrt(np.mean(np.square(NTilde), axis=1))

        # Mean across channels
        NLrmsB = np.mean(NLrmsB, axis=0)
        return NLrmsB

    def loudnessTest(self, Ntot_T, Ntot_R, Ndel=0):
        """Performs the loudness test (page 36).

        Inputs :
        --------
        ``Ntot_T`` : Array-like
            Total noise loudness of the tested signal
        ``Ntot_R`` : Array-like
            Total noise loudness of the reference signal

        Returns :
        ---------
        ``loudnessTest_idx`` : int
            Index of the first frame to fulfill the test.
        """
        Lt = 0.1  # Threshold
        numChannels, numFrames = Ntot_T.shape

        loudnessTest_idx = -1
        loudEnough = False

        while (not loudEnough) and (loudnessTest_idx < numFrames):
            loudnessTest_idx += 1
            for c in range(numChannels):
                if (
                    Ntot_T[c, loudnessTest_idx] >= Lt
                    and Ntot_R[c, loudnessTest_idx] >= Lt
                ):
                    loudEnough = True

        # Adding the additional delay of 50ms Noff
        loudnessTest_idx = self.N50ms  + loudnessTest_idx
        loudnessTest_idx = max(loudnessTest_idx, Ndel)

        return loudnessTest_idx

    def compute_bandwidth(self, X_T, X_R):
        """
        Computation of the bandwidth of the reference and test signals. See [1] section 5.4, page 40.

        Inputs:
        -------
        ``X_T``: Array-like
            STFT of the test signal
        ``X_R``: Array-like
            STFT of the reference signal

        Returns:
        --------
        ``W_R``: float
            bandwidth of the reference signal
        ``W_T``: float
            bandwidth of the test signal
        """

        numChannels, numBins, numFrames = X_T.shape

        # Decibel scale
        X2_T = np.square(X_T)
        X2_R = np.square(X_R)

        # Finding the threshold
        higherFreq_hz = 21586
        higherFreq_idx = int(self.NF * higherFreq_hz / self.sr_hz)
        thresholdLevel = np.max(X2_T[:, higherFreq_idx:-1, :], axis=1)

        # Finding KR and KT
        start_idx = np.ones((numChannels, numFrames), dtype=np.int16) * higherFreq_idx
        KR = self.bandwidthSearch(
            X=X2_R, threshold=thresholdLevel * self.idB10(10), start_idx=start_idx - 1
        )
        KT = self.bandwidthSearch(
            X=X2_T, threshold=thresholdLevel * self.idB10(5), start_idx=KR - 1
        )
        # computing the means
        weights = np.where(KR > 347, 1.0, 0.0)
        W_R = np.sum(KR * weights, axis=1) / np.sum(weights, axis=1)
        weights = np.where(KT > 0, 1.0, 0.0) * weights
        W_T = np.sum(KT * weights, axis=1) / np.sum(weights, axis=1)

        ## Mean across channels
        W_R = np.mean(W_R)
        W_T = np.mean(W_T)
        return W_R, W_T

    @staticmethod
    @njit
    def bandwidthSearch(
        X: npt.ArrayLike,
        threshold: float,
        start_idx: npt.ArrayLike,
    ):
        """
        Searches the 1st bin exceeding the ``threshold`` value.
        Used in compute_bandwidth

        Inputs:
        -------
        - ``X``: Array-like (c, F, T)
            Magnitude spectrogram
        - ``threshold``: float
            Threshold
        - ``start_idx``: Array-like (c, T)
            array of starting frequency indices.
        """
        numChannels, _, numFrames = X.shape

        bandwidth_idx = np.zeros((numChannels, numFrames), dtype=np.int32)
        for chan_idx in range(numChannels):
            for frame_idx in range(numFrames):
                bin_idx = start_idx[chan_idx, frame_idx]
                while (
                    X[chan_idx, bin_idx, frame_idx] < threshold[chan_idx, frame_idx]
                    and bin_idx > 0
                ):
                    bin_idx -= 1
                if bin_idx == 0:
                    bin_idx = -1
                bandwidth_idx[chan_idx, frame_idx] = bin_idx

        return bandwidth_idx + 1

    def get_maskThreshold(self):
        m_dB = 3 * np.ones((self.numBarkBands))
        k = np.arange(self.numBarkBands)
        idx = np.where(k > 12 / self.barkWidth)

        m_dB[idx] = 0.25 * k[idx] * self.barkWidth
        return m_dB

    def masking(
        self, EsTilde_T: npt.ArrayLike, EsTilde_R: npt.ArrayLike, EbN: npt.ArrayLike
    ):
        numChannels, numBins, numFrames = EbN.shape
        # Masking threshold, Eq. (114)
        M = EsTilde_R * self.gm

        # Noise to mask ratio, Eq (116)
        RNM = EbN / M

        ## Total NMR
        # Eq. (117)
        RNMtot = self.dB10(np.mean(RNM, axis=(1, 2)))

        ## Relative disturbed frames
        # Eq. (118)
        RNmax = np.amax(RNM, axis=1)

        RNmax_dB = self.dB10(RNmax)

        RelDistFrames = np.mean(np.asarray(RNmax_dB > 1.5), axis=1)

        # Mean across channels

        RNMtot = np.mean(RNMtot)

        RelDistFrames = np.mean(RelDistFrames)

        return RNMtot, RelDistFrames

    def detectionProbability(self, EsTilde_T, EsTilde_R):
        numChannels, numBins, numFrames = EsTilde_T.shape

        # Converting to dBs, Eq. (121)
        EsTilde_T_dB = self.dB10(EsTilde_T)
        EsTilde_R_dB = self.dB10(EsTilde_R)
        EdB = EsTilde_R_dB - EsTilde_T_dB

        # Asymmetric excitation, Eq. (120)
        L = np.where(
            EsTilde_R > EsTilde_T,
            0.3 * EsTilde_R_dB + 0.7 * EsTilde_T_dB,
            EsTilde_T_dB,
        )

        # Detection step size, Eq (122)
        c0 = -0.198719
        c1 = 0.0550197
        c2 = -0.00102438
        c3 = 5.05622e-6
        c4 = 9.01033e-11

        d1 = 5.95072
        d2 = 6.39468
        gamma = 1.71332

        # Eq. (122)
        s = c0 + L * (c1 + L * (c2 + L * (c3 + L * c4))) + d1 * np.power(d2 / L, gamma)
        s = np.where(L > 0, s, 1e30)

        # Steepness of slope, Eq. (124)
        b = np.where(EsTilde_R > EsTilde_T, 4, 6)
        # Probability of detection, Eq. (123)
        pc = 1 - np.power(0.5, np.power(((EdB) / s), b))

        # Number of steps above the threshold, Eq. (125)
        qc = np.abs((EdB).astype(np.int32)) / s

        # Total probability of detection, Eqs (126) and (127)
        Pb = 1 - np.prod(1 - np.amax(pc, axis=0), axis=0)
        Qb = np.sum(np.amax(qc, axis=0), axis=0)

        ## MFPD_B Computations
        # Filtered probability of detection, Eq. (128)
        c0 = 0.9
        PbTilde = scipy.signal.lfilter(b=[1 - c0], a=[1, -c0], x=Pb)

        # Maximum filtered probability of detection, Eqs. (129) and (130)
        PM = np.amax(PbTilde)
        MFPD_B = PM

        ## ADB_B Computations
        # Total number of steps above the threshold
        indices = np.where(Pb > 0.5)
        Qs = np.sum(Qb[indices], axis=0)

        N = len(indices[0])  # Number of values for which Pb>0.5

        if N == 0:
            ADB_B = 0
        elif Qs > 0:
            ADB_B = np.log10(Qs / N)
        else:
            ADB_B = -0.5

        return MFPD_B, ADB_B

    @staticmethod
    def dot(Di, Dl):
        return np.sum(Di * Dl, axis=1)

    @staticmethod
    def squaredNorm(Di):
        return np.sum(np.abs(np.square(Di)), axis=1)

    def errorHarmonicStructure(
        self,
        X_T: npt.ArrayLike,
        X_R: npt.ArrayLike,
        x_T: npt.ArrayLike,
        x_R: npt.ArrayLike,
    ):
        numChannels, numBands, numFrames = X_T.shape
        # Difference weighted log spectra, Eq. (133)

        Lmax = 256
        NL = Lmax
        M = Lmax

        L = 512

        kmax = NL+M-1
        threshold_idx = self.find_energyThreshold(x_T=x_T, x_R=x_R, X_T=X_T)

        # D = np.zeros((numChannels, kmax, numFrames))
        # for frame_idx in range(numFrames):
        #     if threshold_idx[:, frame_idx] > 0:
        #         D[:, :, frame_idx] = np.log(
        #             np.square(X_T[:, :kmax, frame_idx] / X_R[:, :kmax, frame_idx])
        #         )

        D = 2 * np.log(X_T[:,:kmax,:] / X_R[:,:kmax,:])

        # Normalized autocorrelation, Eq. (135)
        d0 = np.fft.rfft(D[:, :M, :], n=L, axis=1)
        d = np.fft.rfft(D, n=L, axis=1)
        c = np.conj(d0) * d
        C = np.real(np.fft.irfft(c, axis=1, n=L))[:, :NL, :]
        # C = np.ones((numChannels, Lmax, numFrames))

        D0 = D[:, 0:M, :]
        D0norm = C[:,0,:].copy()
        Dlnorm = D0norm
        C[:,0,:]=1
        for l in range(1, Lmax):
            Dlnorm = (
                Dlnorm + (np.square(D[:, l + M - 1, :]) - np.square(D[:, l - 1, :]))
            )
            
            d=D0norm*Dlnorm
            C[:,l,:] = np.where(d<=0, 1, C[:,l,:]/np.sqrt(d))
            # C[:, l, :] /= np.sqrt(D0norm * Dlnorm)
        
        # Windowed correlation, Eqs. (140), (141), (142)
        # H = self.get_hannWindow(NF=NL).reshape(1, NL, 1)
        H = self.get_hannWindow(NF=NL).reshape(1, NL, 1) / NL

        Cmean = np.mean(C, axis=1, keepdims=True)
        Cw = H * (C - Cmean)

        S = np.square(np.abs(np.fft.rfft(Cw, axis=1, n=NL)))
        # This computation matches the Matlab code but not the report
        # EH_max = np.max(S[:, 1:, :], axis=1)
        # EH_max = np.where(EH_max > S[:, 0, :], EH_max, 0)

        EH_max = np.zeros((numChannels, numFrames))
        for channel_idx in range(numChannels):
            for frame_idx in range(numFrames):

                val_prev = S[channel_idx, 0, frame_idx]
                for n in range(1, NL // 2+1):
                    val = S[channel_idx, n, frame_idx]
                    if val > val_prev:
                        if val > EH_max[channel_idx, frame_idx]:
                            EH_max[channel_idx, frame_idx] = val
                    val_prev = val

        ## Error Harmonic structure with energy threshold
        EHB = np.sum(EH_max * threshold_idx, axis=1) / np.sum(threshold_idx, axis=1)
        EHB *= 1000
        # Mean across channels
        EHB = np.mean(EHB)

        return EHB

    def find_energyThreshold(self, x_T, x_R, X_T):
        # Energy threshold, Eq. (145)
        A2thr = 8000 * np.square(self.Amax / 32768)
        # A2thr=8000
        numChannels, _, numFrames = X_T.shape

        A2_T = np.zeros((numChannels, numFrames))
        A2_R = np.zeros((numChannels, numFrames))

        # energy calculation, Eq. (146)
        for frame_idx in range(numFrames):
            start_idx = frame_idx * self.hopSize + self.hopSize
            end_idx = frame_idx * self.hopSize + self.NF
            A2_T[:, frame_idx] = np.sum(np.square(x_T[:, start_idx:end_idx]), axis=1)
            A2_R[:, frame_idx] = np.sum(np.square(x_R[:, start_idx:end_idx]), axis=1)

        # A2_T = A2_T.reshape(numChannels, numFrames)
        # A2_R = A2_R.reshape(numChannels, numFrames)

        # indices where the signal is below the threshold, Eq. (147)
        idx_T = np.asarray(A2_T > A2thr, dtype=np.int32)
        idx_R = np.asarray(A2_R > A2thr, dtype=np.int32)

        # Condition accross channels
        idx_T = np.max(idx_T, axis=0, keepdims=True)
        idx_R = np.max(idx_R, axis=0, keepdims=True)

        # Condition across signals
        idx = np.maximum(idx_T, idx_R)

        # Now the 1s in idx indicate where both channels of both signals are
        # under the threshold

        # Inverting the index so that the ones indicate where one of the signals
        # is above the threshold
        #idx = 1 - idx
        return idx
