import numpy as np
import numpy.typing as npt
#import numba


class Mixin:
    def __init__(self, output='odg'):
        # The weights and biases are hardcoded as in the report
        self.output=output
        wx = np.array(
            [
                [-0.502657, 0.436333, 1.219602],
                [4.307481, 3.246017, 1.123743],
                [4.984241, -2.211189, -0.192096],
                [0.051056, -1.762424, 4.331315],
                [2.321580, 1.789971, -0.754560],
                [-5.303901, -3.452257, -10.814982],
                [2.730991, -6.111805, 1.519223],
                [0.624950, -1.331523, -5.955151],
                [3.102889, 0.871260, -5.922878],
                [-1.051468, -0.939882, -0.142913],
                [-1.804679, -0.503610, -0.620456],
            ]
        )
        wxb = np.array([-2.518254, 0.654841, -2.207228])
        self.lin1 = Linear(w=wx, b=wxb)

        wy = np.array([[-3.817048], [4.107138], [4.629582]])
        wyb = np.array([-0.307594])
        self.lin2 = Linear(w=wy, b=wyb)

        self.MOVs_min = np.array(
            [
                393.916656,
                361.965332,
                -24.045116,
                1.110661,
                -0.206623,
                0.074318,
                1.113683,
                0.950345,
                0.029985,
                0.000101,
                0,
            ]
        )

        self.MOVs_max = np.array(
            [
                921,
                881.131226,
                16.212030,
                107.137772,
                2.886017,
                13.933351,
                63.257874,
                1145.018555,
                14.819740,
                1.0,
                1.0,
            ]
        )

    @staticmethod
    def scale_var(x, amin, amax):
        return (x - amin) / (amax - amin)

    @staticmethod
    def sigmoid(x: npt.ArrayLike):
        # Asymmetric sigmoid, Eq. (150)
        return 1 / (1 + np.exp(-x))

    def neuralNet(self, MOVs_vect_norm):
        """Compute the ODG from the MOVs vect

        Inputs:
        -------

        ``MOVs_vect`` : array-like (shape [1,11])
        """

        out = self.sigmoid(self.lin1(MOVs_vect_norm))
        out = self.lin2(out)[0]
        return out

    def ODG(self, MOVs_vect):

        MOVs_vect_norm = self.scale_var(MOVs_vect, self.MOVs_min, self.MOVs_max)

        # Distortion index, Eq. (149)
        DI = self.neuralNet(MOVs_vect_norm)
        # Objective grade difference, Eq. (151)
        bmin = -3.98
        bmax = 0.22
        ODG = bmin + (bmax - bmin) * self.sigmoid(DI)

        if self.output == "odg":
            return ODG
        elif self.output == "di":
            return DI
        elif self.output == "full":
            return ODG, DI
        else:
            raise ValueError()


class Linear:
    """
    Linear layer of a neural network.
    """

    def __init__(
        self,
        w: npt.ArrayLike = None,
        b: npt.ArrayLike = None,
    ):
        self.w = w
        self.b = b
        self.n_in, self.n_out = w.shape

    @staticmethod
    def func_linear(x, w, b):
        """
        Computations of a linear layer of a neural network.

        Inputs:
        -------
        ``x`` : array-like of shape n_in
        ``w`` : array-like of shape (n_in, n_out)
        ``b`` : array-like of shape n_out
        """

        out = np.sum(x.reshape(x.shape[0], 1) * w, axis=0)
        return out + b

    def __call__(self, x):
        return self.func_linear(x, w=self.w, b=self.b)


if __name__ == "__main__":
    mix = Mixin()
    x = np.random.rand(11)

    print(mix.lin1.w.shape)
    print(mix.lin1.b.shape)
    print(mix.lin2.w.shape)
    print(mix.lin2.b.shape)

    print(mix.neuralNet(x))
