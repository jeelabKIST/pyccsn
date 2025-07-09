from scipy.signal import correlate
import numpy as np


# Get cross-correlation
def get_correlation(x, y, srate, max_lag=None, norm=True):
    """
    Compute the cross-correlation between two signals.

    Args:
        x (array_like, 1D): First input signal.
        y (array_like, 1D): Second input signal.
        srate (float): Sample rate of the signals.
        max_lag (float, optional): Maximum lag for cross-correlation. Defaults to None.
        norm (bool, optional): Whether to normalize the signals. Defaults to True.

    Returns:
        cc (ndarray): Cross-correlation.
        tlag (ndarray): Time lags
            - Positive values indicate that y leads x.
            - Negative values indicate that x leads y.s
    """
    
    # positive: y leads x
    # negative: x leads y
    
    if norm:
        xn = x - np.average(x)
        yn = y - np.average(y)
        std = [np.std(xn), np.std(yn)]
    else:
        xn, yn = x, y
        std = [1, 1]

    if max_lag is None:
        max_lag = len(x)/srate
    max_pad = int(max_lag * srate)
    tlag = np.arange(-max_lag, max_lag+1/srate/10, 1/srate)

    if (std[0] == 0) or (std[1] == 0):  
        return np.zeros(2*max_pad+1), tlag
    
    pad = np.zeros(max_pad)
    xn = np.concatenate((pad, xn, pad))
    cc = correlate(xn, yn, mode="valid", method="fft")/std[0]/std[1]

    
    num_use = len(yn)
    cc = cc/num_use
    
    return tlag, cc