import numpy as np
import matplotlib.pyplot as plt
from test_all import get_refAndCod

class result():
    def __init__(self, data, marker):
        self.data=data
        self.marker=marker



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

computed_ODG = np.load('computed_ODG.npy')

computed_ODG = result(computed_ODG, marker='x')
ODG_PQeval_C = result(ODG_PQeval_C, marker='+')
ODG_PQeval_expected = result(ODG_PQeval_expected, marker='1')


results_dict = {
    'PyEvalAudio (Ours)':computed_ODG,
    'PQevalAudio (Computed)':ODG_PQeval_C,
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