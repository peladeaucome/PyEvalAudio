import numpy as np
import numpy.typing as npt


class Mixin:
    def dB20(self, x:npt.ArrayLike) -> npt.ArrayLike:
        """
        Converts the input signal to sample-wise decibel scale.
        
        Inputs:
        ----------
        `x` : array-like
            input signal
        
        Returns:
        --------
        `x_dB`: array-like, same size as `x`
            sample_wise decibel values of x.
        """
        return 20*np.log10(np.abs(x))

    def dB10(self, x:npt.ArrayLike) -> npt.ArrayLike:
        """
        Converts the input signal to sample-wise decibel scale.
        
        Inputs:
        ----------
        `x` : array-like
            input signal
        
        Returns:
        --------
        `x_dB`: array-like, same size as `x`
            sample_wise decibel values of x.
        """
        return 10*np.log10(np.abs(x))

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


