import numpy as np
import pandas as pd
import os.path as op
from pprint import pformat
# EEG utilities
import mne
from mne.preprocessing import ICA, create_eog_epochs
from pyprep.prep_pipeline import PrepPipeline
# BIDS utilities
from util.io.bids import DataSink
from bids import BIDSLayout

# filter settings
LOW_CUTOFF = 1.
HIGH_CUTOFF = 70.
# trial rejection criteria
CUTOFF = 150e-6 # volts
# directory stuff
BIDS_ROOT = 'bids_dataset'
SOURCE_WORKFLOW = 'PREP'
DERIV_WORKFLOW = 'clean'
DERIV_ROOT = op.join(BIDS_ROOT, 'derivatives')
TASK = 'daf'


def main(sub, layout):
    '''
    Completes additional data cleaning steps post-PREP pipeline,
    and outputs epoched data.

    Parameters
    ----------
    sub : str
        Subject ID as in BIDS dataset
    layout : pybids.BIDSLayout
    '''

    print('\nStarting subject %s.\n'%sub)

    ## load data
    f = layout.get(subject = sub, scope = 'PREP', extension = 'fif.gz')[0]
    raw = mne.io.read_raw_fif(f.path, preload = True)
    # contruct MNE event array from BIDS event file
    evs = layout.get(subject = sub, suffix = 'events')[0].get_df()
    midpoint = (evs.onset + (evs.onset + evs.duration))/2
    new_dur = evs.duration.median() # clip trials to all be same duration
    onset = (midpoint - new_dur/2).to_numpy()
    onset = np.round(onset * raw.info['sfreq']).astype(int)
    name_to_code = {name: i for i, name in enumerate(evs.trial_type.unique())}
    codes = evs.trial_type.replace(name_to_code).to_numpy().astype(int)
    events = np.stack([onset, np.zeros_like(onset), codes], axis = 1)

    # filter and then epoch the data accordingly
    epochs = mne.Epochs(
        raw.copy().filter(LOW_CUTOFF, HIGH_CUTOFF, picks = ['eeg']),
        events, event_id = name_to_code,
        tmin = 0, tmax = new_dur,
        baseline = None,
        preload = True,
        metadata = evs # keep BIDS event info as metadata
    )

    # compute ICA components from highpass filtered copy of data
    # because ICA performs poorly with low frequency drift,
    # but we still want high frequencies to capture noise components
    ica = ICA(n_components = 60, random_state = 0)
    raw = raw.load_data().filter(l_freq = 1., h_freq = None)
    epo = mne.Epochs(raw, events, tmin = 0, tmax = new_dur, baseline = None)
    ica.fit(epo, picks = 'eeg')
    del epo
    del raw
    # and exclude ICA components that are correlated with EOG or EMG
    emg_indices, emg_scores = ica.find_bads_muscle(epochs, threshold = .75)
    eog_indices, eog_scores = ica.find_bads_eog(epochs, threshold = 1.96)
    exclude = np.unique(eog_indices + emg_indices).tolist()
    epochs = ica.apply(epochs, exclude = exclude)

    # reject trials that exceed (arbitrary) voltage threshold
    epochs = epochs.drop_bad(reject = dict(eeg = CUTOFF))

    # switch to laplacian montage to further attenuate EMG artifact
    epochs = mne.preprocessing.compute_current_source_density(epochs)

    # and save preprocessed data
    sink = DataSink(DERIV_ROOT, DERIV_WORKFLOW)
    fpath = sink.get_path(
        subject = sub,
        task = TASK,
        desc = 'clean',
        suffix = 'epo',
        extension = 'fif.gz'
    )
    epochs.save(fpath)

    # generate a report
    report = mne.Report(verbose = True)
    report.parse_folder(
        op.dirname(fpath), pattern = '*epo.fif.gz',
        render_bem = False, raw_butterfly = False,
        )
    if emg_indices:
        fig_ica = ica.plot_components(emg_indices, show = False)
        report.add_figure(
            fig_ica,
            title = 'Removed EMG Components',
            section = 'ICA'
        )
    if eog_indices:
        fig_ica = ica.plot_components(eog_indices, show = False)
        report.add_figure(
            fig_ica,
            title = 'Removed EOG Components',
            section = 'ICA'
        )
    report.save(op.join(sink.deriv_root, 'sub-%s.html'%sub), overwrite = True)

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
