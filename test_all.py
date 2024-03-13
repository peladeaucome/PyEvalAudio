import librosa
import scipy.io
import peaq_numpy
import os
import numpy as np
import results
import time
import matplotlib.pyplot as plt



def get_refAndCod():
    all_names = [
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

    ref = []
    cod = []

    for name in all_names:
        if "ref" in name:
            ref.append(name.lower())
        else:
            cod.append(name.lower())

    return ref, cod


if __name__=='__main__':
    peaq = peaq_numpy.main.PEAQ(mode="basic", output="full", verbose=False)

    ref_names, cod_names = get_refAndCod()
    
    target_DI = results.get_DI_list()
    target_ODG = results.get_ODG_list()


    num_examples = 16

    computed_DI = np.zeros(num_examples)
    computed_ODG = np.zeros(num_examples)

    for obj_idx in range(num_examples):
        Fref = os.path.normpath(f"Data/Conformance_Test_Items/{ref_names[obj_idx]}")
        Ftest = os.path.normpath(f"Data/Conformance_Test_Items/{cod_names[obj_idx]}")

        # print(Fref, Ftest)

        # x_R, sr = librosa.load(Fref, sr=None, mono=False)
        # x_T, sr = librosa.load(Ftest, sr=None, mono=False)

        sr, x_R = scipy.io.wavfile.read(Fref)
        sr, x_T = scipy.io.wavfile.read(Ftest)

        x_R = x_R.T
        x_T = x_T.T

        ODG, DI = peaq.compute_PEAQ(x_T=x_T, x_R=x_R)
        computed_DI[obj_idx] = DI
        computed_ODG[obj_idx] = ODG

        print(f"Index: {obj_idx}")
        print(f"DI : Expected: {target_DI[obj_idx]}, Computed: {DI}")
        print(f"ODG: Expected: {target_ODG[obj_idx]}, Computed: {ODG}")


    DI_error = computed_DI - target_DI
    ODG_error = computed_ODG - target_ODG

    def AES(input, target=0, confidenceInterval=.95):
        error=input-target
        out = 2*np.sqrt(np.mean(np.square(error/confidenceInterval)))

    def MSE(input, target=0):
        error=input-target
        return np.mean(np.square(error))


    def RMSE(input, target=0):
        return np.sqrt(MSE(input, target))


    def MAE(input, target=0):
        error=input-target
        return np.mean(np.abs(error))


    print("\n")
    print(f"RMSE DI {RMSE(DI_error)}")
    print(f"RMSE ODG {RMSE(ODG_error)}")


    print(f"MAE DI {MAE(DI_error)}")
    print(f"MAE ODG {MAE(ODG_error)}")

    np.save('computed_ODG', computed_ODG)