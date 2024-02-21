import librosa
import scipy.io
import peaq_numpy
import os
import numpy as np

peaq = peaq_numpy.main.PEAQ(mode="basic", Amax=1)

Fref = os.path.normpath("Data/48000/AnotherDayCalling_UnmasteredWAV.wav")
Ftest = os.path.normpath("Data/48000/AnotherDayCalling_Full_Preview.wav")

print(Fref, Ftest)

# sr, x_R = librosa.load(Fref, sr=None)
# sr, x_T = librosa.load(Ftest, sr=None)
sr, x_R = scipy.io.wavfile.read(Fref)
sr, x_T = scipy.io.wavfile.read(Ftest)

x_R = x_R.astype(np.float64)
x_T = x_T.astype(np.float64)

x_R /= 2147483392
x_T /= 2147483392

x_R = x_R.reshape(1, -1)
x_T = x_T.reshape(1, -1)
ODG = peaq.compute_PEAQ(x_T=x_T, x_R=x_R)

print(f'ODG: {ODG}')
