import numpy as np
import os
experiment_name = "CompDec_fast_-25"

ODG_list = np.load(os.path.normpath(f'results_fx/{experiment_name}_ODG.npy'))
MMS_list = np.load(os.path.normpath(f'results_fx/{experiment_name}_MMS.npy'))
MMS_list = np.maximum(0, np.minimum(100, MMS_list))

print(experiment_name)
print(f"Mean ODG: {ODG_list.mean()} +- {ODG_list.std()}")
print(f"Mean MMS: {MMS_list.mean()} +- {MMS_list.std()}")