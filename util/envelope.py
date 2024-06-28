from scipy.signal import savgol_filter as sgf
import pyfar as pf
import numpy as np

def d_savgol_filter(x, Fs, h_freq):
    '''
    get first derivative of smoothed envelope
    '''
    # determine window length from approximate high frequency cutoff
    win_len = (int(np.round(Fs / h_freq)) // 2) * 2 + 1
    # get second derivative of filtered signal
    dx = sgf(x, window_length = win_len, polyorder = 5, deriv = 1)
    return dx

def compute_envelope(aud, fs, h_freq = 30):
    '''
    Parameters
    ----------
    aud : an (n_times,) np.array
    fs : float
        The sammpling frequency
    h_freq : float, default: 30
        Approximate high frequency cutoff to use for Savitzky-Golay filter
        when estimating the time derivative of the envelope

    Returns
    ---------
    envelope : an (n_times,) np.array
        The amplitude envelope of `aud` between 20-5000 Hz estimated
        using a gammatone filterbank that approximates the frequency
        response of the human cochlea.
    onset : an (n_times,) np.array
        A positively valued time series that is high when there is
        an increase in the amplitude of any frequency represented in
        the gammatone filterbank. In other words, this marks the
        onset of acoustic events.
    '''
    # compute amplitude envelope within each equivalent rectangular bandwidth (ERB)
    GFB = pf.dsp.filter.GammatoneBands([30, 5000], sampling_rate = fs, resolution = 1)
    sig = pf.Signal(aud, sampling_rate = fs)
    real, imag = GFB.process(sig)
    envelope = np.abs(real.time + 1j * imag.time)
    # clipped first derivatives within ERBs represent acoustic event onsets
    onset = d_savgol_filter(envelope, fs, h_freq = h_freq).clip(0)
    onset = onset * (envelope.max(1) / onset.max(1))[:, np.newaxis] # scale
    # sum both time series across ERBs
    return envelope.sum(0), onset.sum(0)
