from scipy.signal import correlate, correlation_lags
from scipy.stats import pearsonr
from mne.decoding import Scaler
import mne
import numpy as np
import pandas as pd
import pingouin as pg

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
    X = Scaler(scalings = 'median').fit_transform(X)
    y = epochs[condition].get_data(['csd']) # current source density of EEG
    y = Scaler(scalings = 'median').fit_transform(y)
    emg = epochs[condition].get_data(['emg'])
    emg = Scaler(scalings = 'median').fit_transform(emg)
    X, y, emg = X.transpose(2, 0, 1), y.transpose(2, 0, 1), emg.transpose(2, 0, 1)
    return X, y, emg

def _perm_score_encoding(scorer_, y, y_pred, n_outputs, seed = None):
    # shuffle trials if not first 'permutation', which is observed
    rng = np.random.default_rng(seed)
    n_trials = y.shape[1]
    idx = np.arange(n_trials)
    if seed != 0:
        rng.shuffle(idx)
    y = y[:, idx, :]
    # Re-vectorize and call scorer
    y = y.reshape([-1, n_outputs], order = 'F')
    y_pred = y_pred.reshape([-1, n_outputs], order = 'F')
    assert y.shape == y_pred.shape
    scores = scorer_(y, y_pred, multioutput = 'raw_values')
    return scores

def perm_score_encoding(rf, X, y, n_permutations = 10000, n_jobs = -1):
    '''
    generates permutation distribution for encoding model scores
    '''
    from mne.decoding.receptive_field import _SCORERS
    from mne.parallel import parallel_func
    # get scorer object
    scorer_ = _SCORERS[rf.scoring]
    # Generate predictions, then reshape so we can mask time
    X, y = rf._check_dimensions(X, y, predict = True)[:2]
    n_times, n_epochs, n_outputs = y.shape
    y_pred = rf.predict(X)
    y_pred = y_pred[rf.valid_samples_]
    y = y[rf.valid_samples_]
    # and compute score for every permutation
    parallel, p_func, n_jobs = parallel_func(
        _perm_score_encoding, n_jobs,
        verbose = True
    )
    out = parallel(
        p_func(scorer_, y, y_pred, n_outputs, seed)
        for seed in range(n_permutations)
    )
    return np.stack(out, axis = 0)

def mediation_score(yhat_eeg, yhat_emg, feature, valid_samps):
    '''
    Arguments
    -----------
    yhat_eeg : an (n_times, n_epochs, 1) np.array
        feature values predicted from EEG
    yhat_emg : an (n_times, n_epochs, 1) np.array
        feature values predicted from EMG
    feature : an (n_times, n_epochs) np.array
        The true feature values you're trying to predict
    valid_samps : slice or other valid numpy index
        The samples that are valid for correlation measures,
        as given by `ReceptiveField.valid_samples_`

    Returns
    ----------
    res : dict
        Contains `score`, the correlation between EEG prediction and
        the predicted feature, as well as `total`, `direct`, and `indirect`
        coefficients from mediation analysis, with EMG predictions as mediator.
    '''
    df = pd.DataFrame(
        dict(
            eeg = yhat_eeg[valid_samps].flatten(),
            emg = yhat_emg[valid_samps].flatten(),
            y = feature[valid_samps].flatten()
        )
    )
    # n_boot > 1 to prevent warning, but we don't actually want to do bootstrap
    # since observations aren't independent anyway...
    res = pg.mediation_analysis(data = df, x='eeg', y='y', m='emg', n_boot=2)
    # only return point estimates cause bootstrap intervals aren't valid
    return dict(
        score = pearsonr(df.eeg, df.y).statistic,
        beta_total = res[res.path == 'Total'].loc[:, 'coef'].iloc[0],
        beta_direct = res[res.path == 'Direct'].loc[:, 'coef'].iloc[0],
        beta_indirect = res[res.path == 'Indirect'].loc[:, 'coef'].iloc[0]
    )

def _perm_score(yhat_eeg, yhat_emg, feature, valid_samps, seed = None):
    '''
    computes scores with EEG trials shuffled
    '''
    rng = np.random.default_rng(seed)
    n_trials = yhat_eeg.shape[1]
    idx = np.arange(n_trials)
    if seed != 0: # first permutation is observed values
        rng.shuffle(idx)
    return mediation_score(yhat_eeg[:, idx], yhat_emg, feature, valid_samps)

def perm_score_decoding(yhat_eeg, yhat_emg, feature, valid_samps,
                n_permutations = 10000, n_jobs = -1):
    '''
    Parameters
    -------------
    yhat_eeg : an (n_times, n_epochs, 1) np.array
        feature values predicted from EEG
    yhat_emg : an (n_times, n_epochs, 1) np.array
        feature values predicted from EMG
    feature : an (n_times, n_epochs) np.array
        The true feature values you're trying to predict
    valid_samps : slice or other valid numpy index
        The samples that are valid for correlation measures,
        as given by `ReceptiveField.valid_samples_`
    n_permutations : int
        Number of permutations to generate
    n_jobs : int, default use all
        Number of CPU cores over which to distribute work
    '''
    from mne.parallel import parallel_func
    parallel, p_func, n_jobs = parallel_func(
        _perm_score, n_jobs,
        verbose = True
    )
    out = parallel(
        p_func(yhat_eeg, yhat_emg, feature, valid_samps, seed)
        for seed in range(n_permutations)
    )
    return pd.DataFrame(out)

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
        The lag of the maximum cross-correlation, in seconds. Sign is such that
        a positive value denotes that the prediction of `rf` is delayed relative
        to the time series it is trying to predict.
    '''
    # load data and pull out feature to predict
    features, eeg, _ = get_data(epochs, condition)
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
