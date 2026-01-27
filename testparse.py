# import librosa
import scipy.io
import PyEvalAudio
import os
import numpy as np
import results
import time
import argparse
import matplotlib.pyplot as plt


def init_parser():
    parser=argparse.ArgumentParser()
    parser.add_argument('--pre', type=str)
    parser.add_argument('--post', type=str)
    return parser





if __name__=='__main__':
    peaq = PyEvalAudio.main.PEAQ(mode="basic", output="full", verbose=True)


    target_DI = results.get_DI_list()
    target_ODG = results.get_ODG_list()


    parser = init_parser()
    args = parser.parse_args()

    Fref = os.path.normpath(f"Data/Conformance_Test_Items/{args.pre}ref{args.post}.wav")
    Ftest = os.path.normpath(f"Data/Conformance_Test_Items/{args.pre}cod{args.post}.wav")

    print(f"Ref:  {Fref}\nTest: {Ftest}")

    # x_R, sr = librosa.load(Fref, sr=None, mono=False)
    # x_T, sr = librosa.load(Ftest, sr=None, mono=False)

    sr, x_R = scipy.io.wavfile.read(Fref)
    sr, x_T = scipy.io.wavfile.read(Ftest)
    x_R = x_R.T
    x_T = x_T.T

    print(x_R.shape)

    # plt.plot(x_R[0])
    # plt.plot(x_T[0])
    # plt.show()

    ODG, DI = peaq.compute_PEAQ(x_T=x_T, x_R=x_R)
    print(f'ODG: {ODG}')