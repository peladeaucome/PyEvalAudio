import numpy as np
import musdb
import os
import audio_effects
import PyEvalAudio
import utils
from dataclasses import dataclass
from joblib import Parallel, delayed
def dB20(x, eps=1e-10):
    return 20*np.log(np.abs(np.maximum(x, eps)))

def idB20(x_dB):
    return np.power(10, x_dB/20)

@dataclass
class Experiment:
    name: str
    effects: audio_effects.AudioEffect

def RMS_and_peak_norm(x1, x2):
    x1 = x1/np.mean(np.std(x1,axis=1))
    x2 = x2/np.mean(np.std(x2,axis=1))

    max_val = max(np.max(np.abs(x1)), np.max(np.abs(x2)))

    return x1/max_val, x2/max_val

def f(idx, fx):
    #print(idx)
    mus_item = mus[idx]
    x = mus_item.audio.T
    x = np.mean(x, axis=0)
    
    end_idx=30*44100
    N = x.shape[0]
    x = x[:min(N, end_idx)]
    
    x = x /np.max(np.abs(x))
    
    x /= idB20(10)


    y = fx(x)

    x = x.reshape(1, -1)
    y = y.reshape(1, -1)

    x = RS(x)
    y = RS(y)

    x, y = RMS_and_peak_norm(x, y)
    ODG, MMS = peaq.compute_PEAQ_2fmodel(x_T=y, x_R=x)

    return ODG, MMS

if __name__=='__main__':

    experiment_list = []


    # experiment_list.append(
    #     Experiment(
    #         name="CompBra_fast_-15",
    #         effects=audio_effects.compressor.CompressorBranching(
    #             threshold_dB=-15, ratio=4, attackTime_ms=.5, releaseTime_ms=100, knee_dB=0, samplerate=44100
    #         )
    #     )
    # )

    # experiment_list.append(
    #     Experiment(
    #         name="CompBra_fast_-20",
    #         effects=audio_effects.compressor.CompressorBranching(
    #             threshold_dB=-20, ratio=4, attackTime_ms=.5, releaseTime_ms=100, knee_dB=0, samplerate=44100
    #         )
    #     )
    # )

    # experiment_list.append(
    #     Experiment(
    #         name="CompBra_fast_-25",
    #         effects=audio_effects.compressor.CompressorBranching(
    #             threshold_dB=-25, ratio=4, attackTime_ms=.5, releaseTime_ms=100, knee_dB=0, samplerate=44100
    #         )
    #     )
    # )

    # experiment_list.append(
    #     Experiment(
    #         name="CompDec_fast_-15",
    #         effects=audio_effects.compressor.CompressorDecoupled(
    #             threshold_dB=-15, ratio=4, attackTime_ms=.5, releaseTime_ms=100, knee_dB=0, samplerate=44100
    #         )
    #     )
    # )

    # experiment_list.append(
    #     Experiment(
    #         name="CompDec_fast_-20",
    #         effects=audio_effects.compressor.CompressorDecoupled(
    #             threshold_dB=-20, ratio=4, attackTime_ms=.5, releaseTime_ms=100, knee_dB=0, samplerate=44100
    #         )
    #     )
    # )

    # experiment_list.append(
    #     Experiment(
    #         name="CompDec_fast_-25",
    #         effects=audio_effects.compressor.CompressorDecoupled(
    #             threshold_dB=-25, ratio=4, attackTime_ms=.5, releaseTime_ms=100, knee_dB=0, samplerate=44100
    #         )
    #     )
    # )

    experiment_list.append(
        Experiment(
            name="SC_T-15_W5",
            effects=audio_effects.SoftClipper(threshold_dB=-15, knee_dB=5, samplerate=44100),
        )
    )

    experiment_list.append(
        Experiment(
            name="SC_T-20_W5",
            effects=audio_effects.SoftClipper(threshold_dB=-20, knee_dB=5, samplerate=44100),
        )
    )

    experiment_list.append(
        Experiment(
            name="SC_T-25_W5",
            effects=audio_effects.SoftClipper(threshold_dB=-25, knee_dB=5, samplerate=44100),
        )
    )

    experiment_list.append(
        Experiment(
            name="SC_T-15_W0",
            effects=audio_effects.SoftClipper(threshold_dB=-15, knee_dB=0, samplerate=44100),
        )
    )

    experiment_list.append(
        Experiment(
            name="SC_T-20_W0",
            effects=audio_effects.SoftClipper(threshold_dB=-20, knee_dB=0, samplerate=44100),
        )
    )

    experiment_list.append(
        Experiment(
            name="SC_T-25_W0",
            effects=audio_effects.SoftClipper(threshold_dB=-25, knee_dB=0, samplerate=44100),
        )
    )

    # experiment_list.append(
    #     Experiment(
    #         name="HPF_f100_Q071",
    #         effects=audio_effects.equalizer.HighPass(f0=100, Q=.71,samplerate=44100),
    #     )
    # )

    # experiment_list.append(
    #     Experiment(
    #         name="HPF_f200_Q071",
    #         effects=audio_effects.equalizer.HighPass(f0=200, Q=.71,samplerate=44100),
    #     )
    # )


    musdb_path = os.path.normpath("C:/Users/pelad/Documents/Data/musdb18hq")
    mus = musdb.DB(root=musdb_path, is_wav=True)
    num_tracks = len(mus)

    peaq = PyEvalAudio.PEAQ(Amax=1, verbose=False)
    RS = utils.resample.Resampler(in_samplerate=44100, out_samplerate=48000)




    for exp in experiment_list:
        experiment_name = exp.name
        fx = exp.effects

        ODG_list = np.zeros(num_tracks)
        MMS_list = np.zeros(num_tracks)

        print(experiment_name)
        r = Parallel(n_jobs=12)(delayed(f)(idx, fx) for idx in range(len(mus)))
        ODG_list, MMS_list = zip(*r)

        ODG_list = np.array(ODG_list)
        MMS_list = np.array(MMS_list)

        print(f"Mean ODG: {ODG_list.mean()} +- {ODG_list.std()}")
        save_path = os.path.join("results_fx", experiment_name + "_ODG.npy")
        np.save(save_path, ODG_list)
        save_path = os.path.join("results_fx", experiment_name + "_MMS.npy")
        np.save(save_path, MMS_list)
