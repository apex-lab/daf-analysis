from util.trf import to_db, get_data, xcorr_lag, to_evokeds, FEAT_NAMES
from sklearn.linear_model import RidgeCV
import os.path as op
import numpy as np
from mne.decoding import ReceptiveField
import mne
import re
import json

from util.io.bids import DataSink
from bids import BIDSLayout

BIDS_ROOT = 'bids_dataset'
SOURCE_WORKFLOW = 'clean'
DERIV_WORKFLOW = 'TRF'
DERIV_ROOT = op.join(BIDS_ROOT, 'derivatives')
TASK = 'daf'
# limits of encoding model
TMIN, TMAX = -.5, .5

def main(sub, layout):

    print('\nStarting sub-%s...'%sub)

    # load data
    f = layout.get(subject = sub, scope = 'clean', suffix = 'epo')[0]
    epochs = mne.read_epochs(f.path, preload = True)
    # downsample so we're not doing any extra computation
    epochs.resample(2*epochs.info['lowpass'])
    # log-transform envelopes b/c it imporoves EEG prediction
    epochs.apply_function(to_db, picks = ['audio_envelope', 'egg_envelope'])

    # initialize model
    ridge = RidgeCV(
        alphas = np.logspace(-1, 7, 50),
        alpha_per_target = True,
        fit_intercept = True
    )
    rf = ReceptiveField(
        tmin = TMIN, tmax = TMAX,
        sfreq = epochs.info['sfreq'],
        estimator = ridge,
        patterns = True,
        scoring = 'corrcoef',
        n_jobs = -1
    )

    # fit multivariate encoding model excluding EGG onsets
    print('Fitting first encoding model (1/2)...')
    assert(FEAT_NAMES[-1] == 'egg_onsets')
    features, eeg = get_data(epochs, 'random1')
    rf = rf.fit(features[:, :, :-1], eeg)
    features, eeg = get_data(epochs, 'baseline')
    score_without = rf.score(features[:, :, :-1], eeg)
    sink = DataSink(DERIV_ROOT, DERIV_WORKFLOW)
    fpath = sink.get_path(
        subject = sub,
        task = TASK,
        desc = 'encodingWithoutEGGOnsets',
        suffix = 'score',
        extension = 'npy'
    )
    np.save(fpath, score_without, allow_pickle = False)
    # and including EGG onsets
    print('Fitting second encoding model (2/2)...')
    features, eeg = get_data(epochs, 'random1')
    rf = rf.fit(features, eeg)
    features, eeg = get_data(epochs, 'baseline')
    score_with = rf.score(features, eeg)
    fpath = sink.get_path(
        subject = sub,
        task = TASK,
        desc = 'encodingWithEGGOnsets',
        suffix = 'score',
        extension = 'npy'
    )
    np.save(fpath, score_with, allow_pickle = False)
    print('Done with encoding model.')
    print('Moving onto decoding....')

    scores = dict()
    for i, feat in enumerate(FEAT_NAMES):
        print('Decoding %s...'%feat)
        # remove underscore from feature name for BIDS filename
        rep = re.findall(r'_\w', feat)[0]
        feat_name = feat.replace(rep, rep[-1].upper())
        # fit decoding model for each feature
        features, eeg = get_data(epochs, 'random1')
        rf = rf.fit(eeg, features[:, :, i][:, :, np.newaxis])
        # cross-validate
        features, eeg = get_data(epochs, 'baseline')
        score = rf.score(eeg, features[:, :, i][:, :, np.newaxis])[0]
        scores[feat_name] = score
        # and save the temporal response functions
        filters, patterns = to_evokeds(rf, epochs)
        fpath = sink.get_path(
            subject = sub,
            task = TASK,
            desc = '%sFilters'%feat_name,
            suffix = 'ave', # this is the MNE convention for Evoked objects
            extension = 'fif.gz'
        )
        filters.save(fpath, overwrite = True)
        fpath = sink.get_path(
            subject = sub,
            task = TASK,
            desc = '%sPatterns'%feat_name,
            suffix = 'ave',
            extension = 'fif.gz'
        )
        patterns.save(fpath, overwrite = True)

    # save cross-validation scores
    fpath = sink.get_path(
        subject = sub,
        task = TASK,
        desc = 'decoding',
        suffix = 'score',
        extension = 'json'
    )
    with open(fpath, 'w') as f:
        json.dump(scores, f, indent = 4)


    assert(feat == 'egg_onsets') # check to make sure this was last feature fit
    # then get lag between actual and predicted onsets in post-adaption block
    lags = dict()
    lags['pre->post'] = xcorr_lag(rf, epochs, 'random2', feat_index = -1)
    # retrain on post block
    print('Fitting model on post-adaption block...')
    features, eeg = get_data(epochs, 'random2')
    rf = rf.fit(eeg, features[:, :, -1][:, :, np.newaxis])
    # and get lag for pre-adaption block
    lags['post->pre'] = xcorr_lag(rf, epochs, 'random1', feat_index = -1)
    # then save both lags
    fpath = sink.get_path(
        subject = sub,
        task = TASK,
        desc = 'decoding',
        suffix = 'xcorr',
        extension = 'json'
    )
    with open(fpath, 'w') as f:
        json.dump(lags, f, indent = 4)

    # that's it!
    print('Finished sub-%s.\n'%sub)
    return


if __name__ == "__main__":
    layout = BIDSLayout(BIDS_ROOT, derivatives = True)
    subs = layout.get_subjects(scope = SOURCE_WORKFLOW)
    subs.sort(key = int)
    already_done = layout.get_subjects(scope = DERIV_WORKFLOW)
    for sub in subs:
        if sub in already_done:
            continue
        else:
            main(sub, layout)
