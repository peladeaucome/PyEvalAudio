import numpy as np
import matplotlib.pyplot as plt
from test_all import get_refAndCod
import results

class result():
    def __init__(self, data, marker):
        self.data=data
        self.marker=marker

def isig(x):
    return np.log(x / (1 - x))

def ODG_to_DI(ODG):
    bmin = -3.98
    bmax = .22
    DI = (ODG-bmin)/(bmax-bmin)
    return isig(DI)

cod_names = get_refAndCod()[1]
for i in range(len(cod_names)):
    cod_names[i] = cod_names[i][:-4].lower()


plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 8

cm = 1/2.54
ODG_PQeval_C = np.array([
    -1.415,
    -1.048,
    -1.799,
    -0.368,
    -1.240,
    -0.548,
    -2.131,
    -2.444,
    -0.377,
    -3.772,
    0.045,
    -0.834,
    -0.040,
    -2.273,
    -0.845,
    -0.484
])

ODG_PQeval_matlab = np.array([
    -0.681,
    -0.292,
    -1.798,
    -0.367,
    -1.166,
    -0.549,
    -1.772,
    -2.445,
    -0.376,
    -3.772,
    0.045,
    -0.834,
    -0.040,
    -2.267,
    0.048,
    -0.414
])

ODG_PQeval_expected = np.array([
    1.2,
    .5,
    3.2,
    0.7,
    2.1,
    0.8,
    2.4,
    3.5,
    0.8,
    6.0,
    0.,
    1.5,
    0.1,
    3.75,
    -.05,
    0.7
])/6.4*(-4)


ODG_PEAQ = results.get_ODG_list()

computed_ODG = np.load('computed_ODG.npy')

computed_ODG = result(computed_ODG, marker='x')
ODG_PQeval_C = result(ODG_PQeval_C, marker='+')
ODG_PQeval_matlab = result(ODG_PQeval_matlab, marker='+')

ODG_PQeval_expected = result(ODG_PQeval_expected, marker='1')
ODG_PEAQ = result(ODG_PEAQ, marker='x')


results_dict = {
    'PyPEAQ (Ours)':computed_ODG,
    'PQevalAudio (Computed)':ODG_PQeval_matlab,
    'PQevalAudio (Expected)': ODG_PQeval_expected,
}

x = np.arange(16)


resultsNumber = len(results_dict)
plt.figure(figsize=(12*cm, 7*cm))
for idx in range (resultsNumber):
    key = list(results_dict.keys())[idx]
    y = results_dict[key]
    offset_x = idx-(resultsNumber-1)/2
    offset_x*=.1
    plt.plot(x+offset_x,y.data, y.marker,label=key)

plt.legend()
plt.xlim(-1, 16)
plt.ylim(-4.5, 0.5)
plt.grid(axis='y')
plt.xticks(ticks=x, labels=cod_names, rotation=90)
plt.xlabel('Audio item')
plt.ylabel('ODG')


plt.tight_layout()
plt.savefig('Figures/Article/PEAQresultsComparison.pdf')
#plt.show()
plt.close()

def RMSE(x, y=0):
    return np.sqrt(np.mean(np.square(x-y)))

DI_PEAQ = results.get_DI_list()

print(f"ODG RMSE computed:    {RMSE(x=ODG_to_DI(computed_ODG.data), y=DI_PEAQ)}")
print(f"ODG RMSE PQevalAudio: {RMSE(x=ODG_to_DI(ODG_PQeval_matlab.data), y=DI_PEAQ)}")