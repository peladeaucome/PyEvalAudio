import numpy as np
import musdb
import os
import audio_effects
import peaq_numpy
import utils
from dataclasses import dataclass
from joblib import Parallel, delayed
@dataclass
class Experiment():
    name:str
    effects:audio_effects.AudioEffect


experiment_list = []




experiment_list.append(
    Experiment(
        name='LPF8000',
        effects = audio_effects.equalizer.LowPass(f0=8000, Q=0.71, samplerate=44100)
    )
)

experiment_list.append(
    Experiment(
        name='HPF100',
        effects = audio_effects.equalizer.HighPass(f0=100, Q=0.71, samplerate=44100)
    )
)

musdb_path = os.path.normpath("C:/Users/pelad/Documents/Data/musdb18hq")
mus = musdb.DB(root=musdb_path, is_wav=True)
num_tracks = len(mus)

peaq = peaq_numpy.PEAQ(Amax=1, verbose=False)
RS = utils.resample.Resampler(in_samplerate=44100, out_samplerate=48000)

def f(idx, fx):
    print(idx)
    mus_item=mus[idx]
    x = mus_item.audio.T
    x = np.mean(x, axis=0)

    y = fx(x)

    x = x.reshape(1, -1)
    y = y.reshape(1, -1)

    x = RS(x)
    y = RS(y)

    ODG, MMS = peaq.compute_PEAQ_2fmodel(x_T=y, x_R=x)

    return ODG, MMS



for exp in experiment_list:
    experiment_name = exp.name
    fx = exp.effects

    ODG_list = np.zeros(num_tracks)
    MMS_list = np.zeros(num_tracks)

    print(experiment_name)
    r=Parallel(n_jobs=2)(delayed(f)(idx, fx) for idx in range(len(mus)))
    ODG_list, MMS_list = zip(*r)

    ODG_list = np.array(ODG_list)
    MMS_list = np.array(MMS_list)

    print(f"Mean ODG: {ODG_list.mean()} +- {ODG_list.std()}")
    save_path = os.path.join("results", experiment_name + "_ODG.npy")
    # np.save(save_path, ODG_list)
    save_path = os.path.join("results", experiment_name + "_MMS.npy")
    # np.save(save_path, MMS_list)
