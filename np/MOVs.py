import numpy as np
import numpy.typing as npt
import scipy.signal


class Mixin:
    def __init__(self):
        # 5.2.1 Delayed Averaging
        tauDel = 0.5
        self.Ndel = np.ceil(tauDel * self.Fss_fft)

        tauOff = 0.05
        self.Noff = np.ceil(tauOff * self.Fss_fft)

        m_dB = self.get_maskThreshold().reshape(1, self.numBarkBands, 1)
        self.gm = 1 / (self.idB10(m_dB))

    def get_dataBoundary(self, x_T, x_R) -> tuple[int, int]:
        """Finds and returns the data boundaries in sample index"""
        Athr = 200 * self.Amax / 32768

        L = 5  # Number of samples in a window
        Athr_over_L = Athr / L

        start_idx: int = 0
        val_R = np.mean(x_R[:, start_idx : start_idx + L])
        val_T = np.mean(x_T[:, start_idx : start_idx + L])
        # Finding the starting point
        while max(val_R, val_T) < Athr_over_L:
            print(start_idx)
            start_idx += 1
            val_R = np.mean(x_R[:, start_idx : start_idx + L])
            val_T = np.mean(x_T[:, start_idx : start_idx + L])

        end_idx: int = x_T.shape[1] - 1
        val_R = np.mean(x_R[:, end_idx - L : end_idx])
        val_T = np.mean(x_T[:, end_idx - L : end_idx])
        while max(val_R, val_T) < Athr_over_L:
            end_idx -= 1
            val_R = np.mean(x_R[:, end_idx - L : end_idx])
            val_T = np.mean(x_T[:, end_idx - L : end_idx])

        startFrame_idx, endFrame_idx = (
            start_idx // self.hopSize,
            end_idx // self.hopSize,
        )

        return startFrame_idx, endFrame_idx

    def compute_modulationChanges(
        self, M_T, M_R, Ebar_R
    ) -> tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]:
        numChannels, _, N = M_T.shape

        # Delayed averaging
        M_T = M_T[:, :, self.Ndel :]
        M_R = M_R[:, :, self.Ndel :]
        Ebar_R = Ebar_R[:, :, self.Ndel :]

        ## WinModDiff1B
        Mdiff1B = np.abs(M_R - M_R) / (1 + M_R)
        Mdiff1bTilde = 100 / self.numBarkBands * np.sum(Mdiff1B, axis=1)

        L = 4
        Mdiff1bTilde_sqrt = np.sqrt(Mdiff1bTilde)
        intermediate_term = np.zeros(numChannels, N - L + 1)
        for n in range(N - L + 1):
            intermediate_term[:, n] = np.sum(Mdiff1bTilde_sqrt[:, n : n + L], axis=1)
        intermediate_term = np.power(intermediate_term / L, 4)

        MWdiff1B = np.sqrt(np.mean(intermediate_term, axis=1))
        MWdiff1B = np.mean(MWdiff1B)  # Mean across stereo channels

        ## AvgModDiff1B
        W1B = np.sum(Ebar_R / (Ebar_R + 100 * np.power(self.E_IN, 0.3)), axis=1)
        MAdiff1B = np.sum(W1B * Mdiff1bTilde, axis=1)
        MAdiff1B = MAdiff1B / np.sum(W1B, axis=1)
        MAdiff1B = np.mean(MAdiff1B)

        ## AvgModDiff2B
        idx = np.where(M_T < M_R)
        Mdiff2B = (M_T - M_R) / (0.01 + M_R)
        Mdiff2B[idx] = 0.1 * (M_R[idx] - M_T[idx]) / (0.01 + M_R[idx])

        Mdiff2BTilde = 100 / self.numBarkBands * np.sum(Mdiff2B, axis=1)

        W2B = W1B

        MAdiff2B = np.sum(W2B * Mdiff2BTilde, axis=1) / np.sum(
            W2B, axis=1
        )  # Sum accross time
        MAdiff2B = np.mean(MAdiff2B)  # Mean across channels

        return MWdiff1B, MAdiff1B, MAdiff2B

    def compute_partialNoiseLoudness(
        self, EP_T, EP_R, M_R, M_T, alpha, T0, S0
    ) -> npt.ArrayLike:

        beta = np.exp(-alpha * (EP_T - EP_R) / EP_R)

        E_t = self.E_IN.copy()
        E_t = E_t.reshape(1, self.numBarkBands, 1)

        s_R = T0 * M_R + S0
        s_T = T0 * M_T + S0

        E0 = 1

        NL = np.power(E_t / (s_T * E0), 0.23)
        NL = NL * (
            np.power(
                1 + np.maximum(s_T * EP_T - s_R * EP_R, 0) / (E_t + beta * s_R * EP_R),
                0.23,
            )
            - 1
        )

        return NL

    def compute_bandwidth(self, XT, XR):
        """
        Computation of the bandwidth of the reference and test signals. See [1] section 5.4, page 40.

        Inputs:
        -------
        ``XT``: Array-like
            STFT of the test signal
        ``XR``: Array-like
            STFT of the reference signal

        Returns:
        --------
        ``WR``: float
            bandwidth of the reference signal
        ``WT``: float
            bandwidth of the test signal
        """

        numChannels, numBins, numFrames = XT.shape

        # Decibel scale
        XT_dB = self.dB20(XT)
        XR_dB = self.dB20(XR)

        # Finding the threshold
        higherFreq_hz = 21600
        higherFreq_idx = int(self.numFftBands * higherFreq_hz / self.sr_hz)

        threshold_idx = np.argmax(XT_dB[:, higherFreq_idx:, :], axis=1)
        thresholdLevel = XT_dB(threshold_idx + higherFreq_idx)

        # Finding KR and KT
        start_idx = np.ones((numChannels, numFrames), dtype=np.int32) * higherFreq_idx
        KR = self.bandwidthSearch(
            X_dB=XR_dB, threshold=thresholdLevel, gap_dB=10, start_idx=start_idx
        )

        KT = self.bandwidthSearch(
            X_dB=XR_dB, threshold=thresholdLevel, gap_dB=5, start_idx=KR
        )

        # computing the means
        weigths_R = np.where(KR >= 346, x=np.ones_like(KR), y=np.zeros_like(KR))
        WR = np.sum(KR * weigths_R) / np.sum(weigths_R)

        weigths_T = np.where(KR >= 346, x=np.ones_like(KT), y=np.zeros_like(KT))
        WT = np.sum(KT * weigths_T) / np.sum(weigths_T)

        ## Mean across channels
        WR = np.mean(WR)
        WT = np.mean(WT)
        return WR, WT

    def bandwidthSearch(
        self,
        X_dB: npt.ArrayLike,
        threshold_dB: float,
        gap_dB: float,
        start_idx: npt.ArrayLike,
    ):
        """
        Searches the 1st bin exceeding the ``threshold_dB`` value by ``gap_dB`` dBs.
        Used in compute_bandwidth

        Inputs:
        -------
        - ``X_dB``: Array-like (c, F, T)
            Magnitude spectrogram in dBs
        - ``threshold_dB``: float
            Threshold in dB
        - ``gap_dB`` : float
        - ``start_idx``: Array-like (c, T)
            array of starting frequency indices.
        """
        numChannels, numBins, numFrames = X_dB.shape

        bandwidth_idx = np.zeros((numChannels, numFrames), dtype=np.int32)
        for chan_idx in range(numChannels):
            for frame_idx in range(numFrames):
                bin_idx = start_idx[numChannels, numFrames]
                while (
                    X_dB[chan_idx, bin_idx, frame_idx] < threshold_dB + gap_dB
                    and bin_idx > 0
                ):
                    bin_idx -= 1

                bandwidth_idx[chan_idx, frame_idx] = bin_idx

        return bandwidth_idx

    def loudnessTest(self, Ntot_T, Ntot_R):
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

        loudnessTest_idx = 0
        loudEnough = False

        while not loudEnough:
            for c in range(numChannels):
                if (
                    Ntot_T[c, loudnessTest_idx] >= Lt
                    and Ntot_R[c, loudnessTest_idx] >= Lt
                ):
                    loudEnough = True
                else:
                    loudnessTest_idx += 1

        # Adding the additional delay of 50ms Noff
        loudnessTest_idx += self.Noff

        if loudnessTest_idx < self.Ndel:
            loudnessTest_idx = self.Ndel

        return loudnessTest_idx

    def get_maskThreshold(self):
        m_dB = 3 * np.ones((self.numBarkBands))
        k = np.arange(self.numBarkBands)
        idx = np.where(k > 12 / self.barkWidth)

        m_dB[idx] = 0.25 * k[idx] * self.barkWidth
        return m_dB

    def masking(self, EsTilde_T, EsTilde_R, EbN):

        # Masking threshold, Eq. (114)
        M = self.gm * EsTilde_R

        # Noise to mask ration, Eq (116)
        RNM = EbN / M

        ## Total NMR
        # Eq. (117)
        RNMtot = self.log10(np.mean(RNM, axis=(1, 2)))

        ## Relative disturbed frames
        #Eq. (118)
        RN_max = np.amax(RNM, axis=1)


    def compute_RMSNoiseLoud(self, EP_T, EP_R, M_R, M_T, Ntot_T, Ntot_R):
        NL = self.compute_partialNoiseLoudness(
            EP_T, EP_R, M_R, M_T, alpha=1.5, T0=0.15, S0=0.5
        )

        # Removing the start according to the loudness test
        loudnessTest_idx = self.find_loudnessThreshold(Ntot_T, Ntot_R)
        NL = NL[:, :, loudnessTest_idx:]

        # Computing the MOV
        NiLTilde = 24 / self.numBarkBands * np.sum(NL, axis=1)
        NTilde = np.maximum(NiLTilde, 0)

        NLrmsB = np.sqrt(np.mean(np.square(NTilde), axis=1))
        return NLrmsB

    def detectionProbability(self, EsTilde_T, EsTilde_R):
        numChannels, numBins, numFrames = EsTilde_T.size

        EsTilde_T_dB = self.dB10(EsTilde_T)
        EsTilde_R_dB = self.dB10(EsTilde_R)

        # Asymmetric excitation
        L = np.where(
            EsTilde_R > EsTilde_T,
            x=0.3 * EsTilde_R_dB + 0.7 * EsTilde_T_dB,
            y=EsTilde_T_dB,
        )

        # Detection step size
        c0 = -0.198719
        c_list = np.array(
            [
                0.0550197,
                0.00102438,
                5.05622e-6,
                9.01033e-11,
            ]
        ).reshape(1, 1, 1, 4)
        d1 = 5.95072
        d2 = 6.39468
        gamma = 1.71332

        # Eq. (122)
        L = L.reshape(numChannels, numBins, numFrames, 1)
        s = c0 + np.sum(np.power(L, c_list) + d1 * np.power(d2 / L, gamma), 3)
        s = np.where(L > 0, x=s, y=np.power(10, 30))

        # Steepness of slope, Eq. (124)
        b = np.where(EsTilde_R > EsTilde_T, x=4, y=6)
        # Probability of detection, Eq. (123)
        pc = 1 - np.power(0.5, np.power(((EsTilde_R_dB - EsTilde_T_dB) / s), b))

        # Number of steps above the threshold, Eq. (125)
        qc = np.abs((EsTilde_R_dB - EsTilde_T_dB).astype(np.int32)) / s

        # Total probability of detection, Eqs (126) and (127)
        Pb = 1 - np.prod(1 - np.amax(pc, axis=0), axis=0)
        Qb = np.sum(np.amax(qc, axis=0), axis=0)

        ## MFPD_B Computations
        # Filtered probability of detection, Eq. (128)
        c0 = 0.9
        PbTilde = scipy.signal.lfilter(b=1 - c0, a=c0, x=Pb)

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

    def errorHarmonicStructure(self, X_T, X_R):

        # Difference weighted log spectra, Eq. (133)
        D = 2 * np.log(np.abs(X_T) / np.abs(X_R))
