import numpy as np
from scipy.signal import butter, sosfiltfilt, spectrogram


def bandpass_filter(x, srate, fmin, fmax, fo=5, filter="butter"):
    """
    Apply bandpass filter to the signal x.
    
    Parameters:
    - x: input signal
    - srate: sampling rate
    - fmin: minimum frequency
    - fmax: maximum frequency
    - fo: order of the filter
    - filter: type of filter ('butter' or 'cheby1')
    
    Returns:
    - filtered signal
    """
    sos = get_sosfilter([fmin, fmax], srate, fo, filter)
    return sosfiltfilt(sos, x)
        
        
def get_sosfilter(frange: list, srate, fo=5, filter='butter'):
    """Get second-order sections (SOS) for a digital filter.

    Parameters:
        frange (list): Frequency range [fmin, fmax].
        srate (float): Sampling rate.
        fo (int, optional): Order of the filter. Defaults to 5.
        filter (str, optional): Type of filter ('butter' or 'cheby1'). Defaults to 'butter'.

    Returns:
        sos (ndarray): Second-order sections for the digital filter.
    """
    
    if filter == 'butter':
        sos = butter(fo, np.array(frange)/srate*2, btype="bandpass", output="sos", analog=False)
    elif filter == 'cheby1':
        from scipy.signal import cheby1
        sos = cheby1(fo, rp=2, Wn=np.array(frange)/srate*2, btype="bandpass", output="sos", analog=False)
    return sos

