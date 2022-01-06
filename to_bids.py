from mne_bids import (
    BIDSPath,
    write_raw_bids,
    get_anonymization_daysback,
    update_sidecar_json
)
import mne
from philistine.mne import write_raw_brainvision as write_bv
from bids import BIDSLayout

from scipy.signal import lfilter, butter
from scipy.signal._arraytools import axis_reverse

import pandas as pd
import numpy as np
import os
import re
import json
import tempfile

DATA_DIR = 'source' # where our data currently lives
BIDS_DIR = 'bids_dataset' # where we want it to live
EGG_HIGHPASS = 20 # hardware filter of EGG amplifier, in Hz

try: # if bids directory has already been made
    layout = BIDSLayout(BIDS_DIR)
    finished_subjects = layout.get_subjects()
    finished_subjects = [int(s) for s in finished_subjects]
except:
    finished_subjects = [] # no subjects have been bidsified yet

eeg_fnames = os.listdir(os.path.join(DATA_DIR, 'eeg'))
eeg_fnames = [f for f in eeg_fnames if '.vhdr' in f] # filter for .vhdr files

loc_fnames = os.listdir(os.path.join(DATA_DIR, 'captrak'))
loc_fnames = [f for f in loc_fnames if '.bvct' in f]
loc_fnames = {int(re.findall('(\d+).bvct', f)[0]): f for f in loc_fnames}

log_fnames = os.listdir(os.path.join(DATA_DIR, 'logs'))
log_fnames = {int(re.findall('(\d+).tsv', f)[0]): f for f in log_fnames}

for f in eeg_fnames:

    sub = int(re.findall('sub-(\d+).vhdr', f)[0])
    if sub in finished_subjects:
        continue # move onto the next loop iteration

    if sub != 11:
        loc_f = os.path.join(DATA_DIR, 'captrak', loc_fnames[sub])
    else:
        loc_f = None
    log_f = os.path.join(DATA_DIR, 'logs', log_fnames[sub])
    eeg_f = os.path.join(DATA_DIR, 'eeg', f)

    # read EEG file and rename non-EEG channels
    raw = mne.io.read_raw_brainvision(eeg_f, preload = True)
    raw = raw.rename_channels({'Ch33': 'leog', 'Ch64': 'reog',
                            'Aux1': 'egg', 'Aux2': 'audio'})
    raw = raw.set_channel_types({'leog': 'eog', 'reog': 'eog',
                            'egg': 'misc', 'audio': 'misc'})

    # backwards filter to reverse phase offset induced by EGG's hardware filter
    egg = raw._data[np.array(raw.ch_names) == 'egg', :]
    egg = np.squeeze(egg)
    b, a = butter(1, EGG_HIGHPASS, btype = 'highpass', fs = raw.info['sfreq'])
    y = axis_reverse(egg, axis = 0)
    y = lfilter(b, a, y)
    egg = axis_reverse(y, axis = 0)
    raw._data[np.array(raw.ch_names) == 'egg', :] = egg

    # rename EEG channels to 10-20 positions using Brain Vision provided layout file
    layout = mne.channels.read_custom_montage(os.path.join(DATA_DIR, 'AP-96.bvef'))
    mapping = {'Ch%s'%i: layout.ch_names[i] for i in range(len(layout.ch_names))}
    mapping = {key: value for key, value in mapping.items() if key in raw.ch_names}
    raw = raw.rename_channels(mapping)
    raw = mne.add_reference_channels(raw, 'Cz')

    log = pd.read_csv(log_f, sep = '\t')

    events, event_id = mne.events_from_annotations(raw, verbose = False)
    starts = events[events[:,2] < 125] # sentence starts
    ends = events[events[:,2] == 127] # sentence ends
    resps = events[np.logical_and(events[:,2] >=125, events[:,2] < 127)] # y/n

    # make sure we're not missing any triggers
    assert(starts.shape == ends.shape)
    assert(starts.shape == resps.shape)

    # compile event information into a single, BIDS compatible dataframe
    log.delay = log.delay + 4 # adjust for known 4.02 ms passthrough delay
    log.delay = log.delay / 1000 # convert to seconds to be consistent w/ BIDS
    durs = (ends[:,0] - starts[:,0])/raw.info['sfreq'] # sentence durations
    yn = (resps[:,2] == 125).astype(int) # 1 if yes, 0 if no
    log['detected_delay'] = yn
    onsets = starts[:,0]/raw.info['sfreq']
    log.insert(0, 'duration', durs) # these columns have to come first re: BIDS
    log.insert(0, 'onset', onsets)

    # prepare raw to be saved as BIDS (move back from mem to disk)
    temp_dir = tempfile.TemporaryDirectory()
    temp_f = os.path.join(temp_dir.name, 'raw.vhdr')
    write_bv(raw, temp_f, events = False)
    raw = mne.io.read_raw_brainvision(temp_f, preload = False)
    raw = raw.set_channel_types({'leog': 'eog', 'reog': 'eog',
                            'egg': 'misc', 'audio': 'misc'})
    raw.info['line_freq'] = 60.
    raw = raw.set_annotations(None) # don't write original trigger events

    # add subject-specific electrode positions from captrak file
    if sub != 11: # we don't have coordinates for this one
        dig = mne.channels.read_dig_captrak(loc_f)
        raw = raw.set_montage(dig)

    # write data to BIDS directory
    bids_path = BIDSPath(
        subject = '%02d'%sub,
        task = 'daf',
        datatype = 'eeg',
        root = BIDS_DIR
    )
    saved_at = write_raw_bids(
        raw, bids_path = bids_path,
        overwrite = True
    )
    temp_dir.cleanup()

    # update sidecar json with extra fields
    json_fpath = str(saved_at).replace('vhdr', 'json')
    with open(json_fpath, "r") as f:
        desc = json.load(f)
    desc['EEGReference'] = 'Cz'
    desc['EEGGround'] = 'Fpz'
    desc['EEGPlacementScheme'] = 'extended 10-20'
    with open(json_fpath, "w") as f:
        json.dump(desc, f, indent = 4)

    # save custom events file
    events_fpath = str(saved_at).replace('_eeg.vhdr', '_events.tsv')
    log.to_csv(events_fpath, sep = '\t', index = False, na_rep = 'n/a')

    # edit new channels.tsv to correct some factual errors
    ch_fpath = str(saved_at).replace('_eeg.vhdr', '_channels.tsv')
    ch_info = pd.read_csv(ch_fpath, sep = '\t')
    ch_info.low_cutoff[ch_info.name == 'egg'] = EGG_HIGHPASS
    ch_info.units[ch_info.name == 'egg'] = 'n/a'
    ch_info.units[ch_info.name == 'audio'] = 'n/a'
    ch_info.units[ch_info.name == 'Cz'] = ch_info.units[ch_info.name == 'Fz'].iloc[0]
    ch_info.to_csv(ch_fpath, sep = '\t', index = False, na_rep = 'n/a')
