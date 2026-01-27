import numpy as np
import os
import json
import scipy.io.wavfile
import audio_effects
import PyEvalAudio
import utils
from scripting_experiments_multi import RMS_and_peak_norm
import matplotlib.pyplot as plt
msm_path = os.path.normpath("C:/Users/pelad/Documents/Data/mixing_secrets_mastering")


def get_metadata(msm_path):
    fp = open(os.path.join(msm_path, "mixing_secrets_mastering_aligned.json"))
    metadata = json.load(fp=fp)
    fp.close()
    return metadata


def get_audio(msm_path, metadata, idx):
    keys_list = list(metadata.keys())
    patch_md = metadata[keys_list[idx]]

    _, mix_audio = scipy.io.wavfile.read(
        os.path.join(msm_path, "aligned", patch_md["unmastered_file"])
    )
    _, master_audio = scipy.io.wavfile.read(
        os.path.join(msm_path, "aligned", patch_md["mastered_file"])
    )

    return mix_audio, master_audio


if __name__ == "__main__":
    samplerate = 44100
    fft_filter = audio_effects.fftfilt.LowPass()
    metadata = get_metadata(msm_path=msm_path)
    list_keys = list(metadata.keys())

    peaq = PyEvalAudio.PEAQ(verbose=False, Amax=1)
    rs = utils.resample.Resampler(in_samplerate=44100, out_samplerate=48000)

    idx=0

    x, y = get_audio(msm_path=msm_path, metadata=metadata, idx=idx)
    x = x[: int(30 * samplerate)]
    y = y[: int(30 * samplerate)]
    x = fft_filter(x)

    x=x.reshape(1, int(30 * samplerate))
    y=y.reshape(1, int(30 * samplerate))

    x, y = RMS_and_peak_norm(x, y)
    odg, mms = peaq.compute_PEAQ_2fmodel(x_T=y, x_R=x)
    print(odg, mms)

    # plt.plot(x[0])
    # plt.plot(y[0])
    # plt.show()

    print(f'Number of audios: {len(metadata)}')
    for idx in range(len(metadata)):
        print(f'Audio number :{idx+1}/{len(metadata)}')
        print(f'Song:{metadata[list_keys[idx]]["project"]}')
        x, y = get_audio(msm_path=msm_path, metadata=metadata, idx=idx)
        x = x[: int(30 * samplerate)]
        y = y[: int(30 * samplerate)]
        x = fft_filter(x)

        x=x.reshape(1, int(30 * samplerate))
        y=y.reshape(1, int(30 * samplerate))

        x = rs(x)
        y = rs(y)
        
        x, y = RMS_and_peak_norm(x, y)
        odg, mms = peaq.compute_PEAQ_2fmodel(x_T=y, x_R=x)
        print(f'ODG:{round(odg, 4)}, MMS:{round(mms, 2)}\n')
