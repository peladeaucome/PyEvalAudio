import numpy as np
import numpy.typing as npt


class Mixin:
    def __init__(self):
        # The weights and biases are hardcoded as in the report

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
        self.lin1 = Linear(
            w=wx, b=wxb
        )

        wy = np.array([[-3.817048], [4.107138], [4.629582]])
        wyb = np.array([-0.307594])
        self.lin2 = Linear(w=wy, b=wyb)

    @staticmethod
    def scale_var(x, amin, amax):
        return (x - amin) / (amax - amin)

    @staticmethod
    def sigmoid(x: npt.ArrayLike):
        return 1 / (1 + np.exp(-x))

    

    def neuralNet(self, MOVs_vect):
        """Compute the ODG from the MOVs vect

        Inputs:
        -------

        ``MOVs_vect`` : array-like (shape [1,11])
        """

        out = self.sigmoid(self.lin1(MOVs_vect))
        out = self.lin2(out)
        return out




class Linear:
    """
    Linear layer of a neural network.
    """
    def __init__(
        self,
        w:npt.ArrayLike=None,
        b:npt.ArrayLike=None,
    ):
        self.w=w
        self.b=b
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

        out = np.sum(x.reshape(x.shape[0], 1)*w, axis=0)
        return out+b
    
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
