import numpy as np
import numpy.typing as npt


class Mixin:
    def dB20(self, x:npt.ArrayLike, eps:float=1e-12) -> npt.ArrayLike:
        """
        Converts the input signal to sample-wise decibel scale.
        
        Inputs:
        ----------
        `x` : array-like
            input signal
        `eps` : float
            threshold to avoid digital errors, default = 1e-4.
        
        Returns:
        --------
        `x_dB`: array-like, same size as `x`
            sample_wise decibel values of x.
        """
        x_abs = np.maximum(np.abs(x), eps)
        return 20*np.log10(x_abs)

    def dB10(self, x:npt.ArrayLike, eps:float=1e-12) -> npt.ArrayLike:
        """
        Converts the input signal to sample-wise decibel scale.
        
        Inputs:
        ----------
        `x` : array-like
            input signal
        `eps` : float
            threshold to avoid digital errors, default = 1e-4.
        
        Returns:
        --------
        `x_dB`: array-like, same size as `x`
            sample_wise decibel values of x.
        """
        x_abs = np.maximum(np.abs(x), eps)
        return 10*np.log10(x_abs)

    def idB20(self, x_dB:npt.ArrayLike) -> npt.ArrayLike:
        """
        Converts from decibel amplitudes to linear

        Inputs:
        ----------
        `x_dB` : array-like
            decibel values
        
        Returns:
        --------
        `x` : array-like, same size as x.
        linear amplitude values.
        """
        return np.power(10, x_dB/20)

    def idB10(self, x_dB:npt.ArrayLike) -> npt.ArrayLike:
        """
        Converts from decibel amplitudes to squared amplitude

        Inputs:
        ----------
        `x_dB` : array-like
            decibel values
        
        Returns:
        --------
        `x` : array-like, same size as x.
        squared amplitude values.
        """
        return np.power(10, x_dB/10)


