from scipy.signal import correlate, correlation_lags
from mne.decoding import Scaler
import mne
import numpy as np

# if you change these, you'll have to fix places where features are indexed
# by position in the function below (e.g. assuming the two audio features
# come first and the EGG onsets come last)
FEAT_NAMES = ['audio_envelope', 'audio_onsets', 'egg_envelope', 'egg_onsets']

def to_db(x):
    '''
    converts a channel to decibels (i.e. log scale), because log-transformed
    acoustic envelopes predict EEG better: https://doi.org/10.7554/eLife.85012
    '''
    x[x <= 0] = x[x > 0].min() # clip before log transfrom
    x = 10 * np.log10(x)
    return x

def get_data(epochs, condition):
    '''
    Loads and robustly scales the EGG/audio features EEG data from one block,
    reshaped as expected by mne.decoding.ReceptiveField. It is recommended
    to log-transform the EGG and audio envelopes BEFORE running this function.

    Returns
    --------------
    features : an (n_times, n_epochs, n_features) np.array
        Contains the audio envelope, audio onsets, EGG envelope, and EGG onsets.
    eeg : an (n_times, n_epochs, n_electrodes) np.array

    Notes
    --------------
    Since the features are scaled outside of the cross-validation pipeline,
    you should NOT cross-validate within a condition to prevent train-test
    leakage. Cross-validation should be done across conditions.
    '''
    X = epochs[condition].get_data(FEAT_NAMES)
    delays = epochs[condition].metadata.delay
    for i, d in enumerate(delays):
        # delay audio to match what subject (rather than mic) hears
        delay_samps = np.round(d * epochs.info['sfreq']).astype(int)
        x = X[i, :2, :]
        x = np.roll(x, delay_samps, axis = 1)
        x[:, :delay_samps] = 0
        X[i, :2, :] = x # audio features
    X = Scaler(scalings = 'median').fit_transform(X)
    y = epochs[condition].get_data(['csd']) # current source density of EEG
    y = Scaler(scalings = 'median').fit_transform(y)
    X, y = X.transpose(2, 0, 1), y.transpose(2, 0, 1)
    return X, y

def xcorr_lag(rf, epochs, condition, feat_index = -1):
    '''
    Computes the lag at which cross-correlation between predicted and actual
    feature time series (by default, the EGG onsets) in specified condition.

    Parameters
    ------------
    rf : mne.ReceptiveField
        trained decoding model
    epochs : mne.Epochs
    condition : str
    feat_index : int, default: -1
        The acoustic feature to predict.

    Returns
    ------------
    maxlag: float
        The lag of the maximum cross-correlation, in seconds.
    '''
    # load data and pull out feature to predict
    features, eeg = get_data(epochs, condition)
    y = features[:, :, feat_index][:, :, np.newaxis]
    # predict that feature with trained model
    yhat = rf.predict(eeg)
    yhat = yhat[:, :, feat_index][:, :, np.newaxis]
    # remove invalid samples (i.e. those too near edges)
    y = y[rf.valid_samples_]
    yhat = yhat[rf.valid_samples_]
    # reshape
    y = y.reshape([-1, 1], order = 'F')
    yhat = yhat.reshape([-1, 1], order = 'F')
    # calculate cross correlation at many lags
    corrs = correlate(yhat, y)
    lags = correlation_lags(yhat.size, y.size) / epochs.info['sfreq']
    # and return lag of maximum cross correlation
    return lags[np.argmax(corrs)]

def to_evokeds(rf, epochs):
    '''
    pulls filters (i.e. decoding weights) and patterns (i.e. encoding weights)
    out of a temporal response function model
    '''
    info = epochs.copy().pick(['csd']).info
    assert(rf.patterns_.shape[0] == 1)
    patterns = mne.EvokedArray(rf.patterns_[0, :, :], info, tmin = rf.tmin)
    filters = mne.EvokedArray(rf.coef_[0, :, :], info, tmin = rf.tmin)
    return filters, patterns
