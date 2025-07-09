# from scipy.signal import welch, butter
from scipy import signal
import numpy as np


def compute_spectrum(x, fs, window="boxcar", frange=None, axis=-1):
    """
    Estimate the power spectrum of a signal

    Args:
        x (array_like): Input signal
        fs (float): sample rate of the signal
        window (str, optional): Type of window to use. Defaults to "boxcar".
        frange (tuple, optional): Frequency range to consider. Defaults to None.
        axis (int, optional): Axis along which the spectrum is coputed. Defaults to -1.
    """
    f, px = signal.welch(x, fs=fs,
                         window=window, 
                         nperseg=np.shape(x)[axis], noverlap=None, axis=axis, 
                         detrend=None, 
                         return_onesided=True, scaling='spectrum')
    px *= 2
    
    if frange is not None:
        idf = (f >= frange[0]) & (f <= frange[1])
        f = f[idf]
        px = px[idf]
    
    return f, px


def compute_welch_spectrum(x, fs, nperseg=256, noverlap=None, window="hann", axis=-1):
    """
    Estimate the power spectrum of a signal using Welch's method

    Args:
        x (array_like): Input signal
        fs (float): sample rate of the signal
        nperseg (int, optional): Length of each segment for Welch's method. Defaults to 256.
        noverlap (int, optional): Number of points to overlap between segments. Defaults to None.
        axis (int, optional): Axis along which the spectrum is coputed. Defaults to -1.
    """
    f, px = signal.welch(x, fs=fs,
                         nperseg=nperseg, noverlap=noverlap, axis=axis, 
                         window=window,
                         return_onesided=True, scaling='spectrum', detrend=None)
    px *= 2
    return f, px


def compute_spectrogram(x, fs, t=None, wbin_t=1, mbin_t=0.1, frange=None):
    """
    Estimate the power spectrogram by short-term Fourier transform (STFT)

    Args:
        x (array_like): 1D or 2D input signal
                        For 2D signal, the second dimension is assumed to be time.
        fs (float): sample rate of the signal
        t (array_like, optional): Time vector corresponding to the signal. 
                                    If it is not given, it is computed from the signal and sample rate.
        wbin_t (float, optional): Time window for each segment in seconds. Defaults to 1.
        mbin_t (float, optional): Time bin for the spectrogram in seconds. Defaults to 0.1.
        axis (int, optional): Axis along which the spectrum is coputed. Defaults to -1.
    """
    mbin = int(mbin_t * fs)
    wbin = int(wbin_t * fs)
    
    if len(x) < wbin:
        raise ValueError("Length of signal must be greater than the window length.")
    
    x = np.array(x)
    shrink_axis = False
    if len(np.shape(x)) == 1:
        x = np.expand_dims(x, axis=0)
        shrink_axis = True
    
    if t is None:
        t = np.arange(0, len(x)/fs)
    else:
        t = np.asarray(t)
        assert np.shape(x)[1] == len(t), "Length of signal and time vector must match."
        
    nbin_set = np.arange(int(t[0]*fs), int(t[-1]*fs)-wbin, mbin)
    num_f = wbin//2 + 1
    
    pxx = np.zeros((x.shape[0], num_f, len(nbin_set)))
    for i, n0 in enumerate(nbin_set):
        xseg = x[:, n0:n0+wbin]
        f, pxx[:,:,i] = compute_spectrum(xseg, fs=fs, window="hann", axis=1)
        
    if frange is not None:
        idf = (f >= frange[0]) & (f <= frange[1])
        f = f[idf]
        pxx = pxx[:, idf, :]
    
    if shrink_axis:
        pxx = pxx[0]
        
    tp = (nbin_set + wbin/2)/fs
    
    return f, tp, pxx
    
    
    
