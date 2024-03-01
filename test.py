import librosa
import scipy.io
import peaq_numpy
import os
import numpy as np

import time


All_names = [
    "acodsna.wav",
    "arefsna.wav",
    "bcodtri.wav",
    "breftri.wav",
    "ccodsax.wav",
    "crefsax.wav",
    "ecodsmg.wav",
    "erefsmg.wav",
    "fcodsb1.wav",
    "fcodtr1.wav",
    "fcodtr2.wav",
    "fcodtr3.wav",
    "frefsb1.wav",
    "freftr1.wav",
    "freftr2.wav",
    "freftr3.wav",
    "gcodcla.wav",
    "grefcla.wav",
    "icodsna.wav",
    "irefsna.wav",
    "kcodsme.wav",
    "krefsme.wav",
    "lcodhrp.wav",
    "lcodpip.wav",
    "lrefhrp.wav",
    "lrefpip.wav",
    "mcodcla.wav",
    "mrefcla.wav",
    "ncodsfe.wav",
    "nrefsfe.wav",
    "scodclv.wav",
    "srefclv.wav",
]


def get_refAndCod(all_names):
    ref = []
    cod = []

    for name in all_names:
        if "ref" in name:
            ref.append(name.upper())
        else:
            cod.append(name.upper())

    return ref, cod


ref_names, cod_names = get_refAndCod(All_names)


obj_idx = 9


Fref = os.path.normpath(f"Data/CONFORMANCE TEST ITEMS/{ref_names[obj_idx]}")
Ftest = os.path.normpath(f"Data/CONFORMANCE TEST ITEMS/{cod_names[obj_idx]}")

print(Fref, Ftest)

x_R, sr = librosa.load(Fref, sr=None, mono=False)
x_T, sr = librosa.load(Ftest, sr=None, mono=False)
# sr, x_R = scipy.io.wavfile.read(Fref)
# sr, x_T = scipy.io.wavfile.read(Ftest)
# x_R = np.array(np.matrix(x_R).T).astype(np.float64)
# x_T = np.array(np.matrix(x_T).T).astype(np.float64)


# x_R /= 2147483648
# x_T /= 2147483648
# x_R = x_R.reshape(x_R.shape[1], x_R.shape[0])
# x_T = x_T.reshape(x_T.shape[1], x_T.shape[0])


print(x_R.shape)
peaq = peaq_numpy.main.PEAQ(mode="basic", Amax=1)

tic = time.time()
for i in range(1):
    ODG = peaq.compute_PEAQ(x_T=x_T, x_R=x_R)
toc = time.time()


print(f"ODG: {ODG}")
print(f"Time: {round(toc-tic, 4)}")
