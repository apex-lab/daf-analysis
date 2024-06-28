import numpy as np
import pandas as pd
import os.path as op
from pprint import pformat
# EEG utilities
import mne
from mne.preprocessing import ICA, create_eog_epochs
from pyprep.prep_pipeline import PrepPipeline
# BIDS utilities
from mne_bids import BIDSPath, read_raw_bids
from util.io.bids import DataSink
from util.envelope import compute_envelope
from bids import BIDSLayout

BIDS_ROOT = 'bids_dataset'
DERIV_WORKFLOW = 'PREP'
DERIV_ROOT = op.join(BIDS_ROOT, 'derivatives')
TASK = 'daf'

def add_envelope_channels(raw, chan):
    # compute amplitude envelope and spectral event onsets
    aud, t = raw.get_data(picks = [chan], return_times = True)
    aud = np.squeeze(aud)
    envelope, onsets = compute_envelope(aud, raw.info['sfreq'])
    # create dummy Raw object so we can marge them back into raw
    info = raw.copy().pick(['egg', 'audio']).info
    new_chans = mne.io.RawArray(np.stack([envelope, onsets]), info)
    new_chans = new_chans.rename_channels({
        'egg': '%s_envelope'%chan,
        'audio': '%s_onsets'%chan
        })
    # merge and return
    raw = raw.add_channels([new_chans])
    return raw

def main(sub, layout):
    '''
    Parameters
    ----------
    sub : str
        Subject ID as in BIDS dataset
    layout : pybids.BIDSLayout
    '''
    print('\nStarting subject %s.\n'%sub)
    sub_num = int(sub)
    np.random.seed(sub_num)
    ## load EEG + EGG data
    bids_path = BIDSPath(
        root = BIDS_ROOT,
        subject = sub,
        task = TASK,
        datatype = 'eeg'
        )
    raw = read_raw_bids(bids_path, verbose = False)
    raw.info['bads'] = [] #################### or PREP will fail later!
    # this is workaround for https://github.com/sappelhoff/pyprep/issues/146
    ## load experiment events
    f = layout.get(
        return_type = 'file',
        subject = sub,
        suffix = 'events',
        extension = 'tsv'
        )[0]
    evs = pd.read_csv(f, sep = '\t')

    raw.load_data() # load data and highpass audio to match EGG
    df = layout.get(subject = sub, task = TASK, suffix = 'channels')[0].get_df()
    egg_lfreq = df[df.name == 'egg']['low_cutoff'].iloc[0]
    raw = raw.filter(egg_lfreq, None, picks = ['audio'])
    # add amplitude envelope and spectral event onsets for EGG/audio channels
    # (must be done before downsampling for PREP b/c requires high frequencies)
    raw = add_envelope_channels(raw, 'audio')
    raw = add_envelope_channels(raw, 'egg')

    # re-reference eye electrodes to become bipolar EOG
    def ref(dat):
        dat[0,:] = (dat[0,:] - dat[1,:])
        return dat
    raw = raw.apply_function(ref, picks = ['leog', 'Fp2'], channel_wise = False)
    raw = raw.apply_function(ref, picks = ['reog', 'Fp1'], channel_wise = False)
    raw = raw.set_channel_types({'leog': 'eog', 'reog': 'eog'})

    # run PREP pipeline (notch, exclude bad chans, and re-reference)
    raw = raw.resample(1000.) # downsample for PREP as a workaround for
    lf = raw.info['line_freq'] # pyprep Issue #109 on GitHub:
    prep_params = {            # https://github.com/sappelhoff/pyprep/issues/109
        "ref_chs": "eeg",
        "reref_chs": "eeg",
        "line_freqs": np.arange(lf, 200., lf)
    }
    raw = raw.set_eeg_reference('average')
    prep = PrepPipeline(
        raw,
        prep_params,
        raw.get_montage(),
        ransac = True, 
        random_state = int(sub)
        )
    prep.fit()
    raw = prep.raw_eeg.add_channels(
        [prep.raw_non_eeg],
        force_update_info = True
    )
    raw.info['bads'] = [] # already interpolated by PREP

    # create bipolar jaw EMG
    raw = raw.apply_function(ref, picks = ['FT9', 'TP9'], channel_wise = False)
    raw = raw.apply_function(ref, picks = ['FT10', 'TP10'], channel_wise=False)
    raw = raw.rename_channels({'FT9': 'lemg', 'FT10': 'remg'})
    raw = raw.set_channel_types({'lemg': 'emg', 'remg': 'emg'})

    # and save minimally preprocessed data
    sink = DataSink(DERIV_ROOT, DERIV_WORKFLOW)
    fpath = sink.get_path(
        subject = sub,
        task = TASK,
        desc = 'PREP',
        suffix = 'raw',
        extension = 'fif.gz'
    )
    raw.save(fpath)

    # generate a report
    report = mne.Report(verbose = True, raw_psd = True)
    report.parse_folder(
        op.dirname(fpath), pattern = '*raw.fif.gz',
        render_bem = False, raw_butterfly = False,
        )
    bads = prep.noisy_channels_original
    html_lines = []
    for line in pformat(bads).splitlines():
        html_lines.append('<br/>%s' % line)
    html = '\n'.join(html_lines)
    report.add_html(html, title = 'Interpolated Channels')
    report.add_html(raw.info._repr_html_(), title = 'Info')
    report.save(op.join(sink.deriv_root, 'sub-%s.html'%sub), overwrite = True)

if __name__ == "__main__":
    layout = BIDSLayout(BIDS_ROOT, derivatives = True)
    subs = layout.get_subjects()
    subs.sort(key = int)
    already_done = layout.get_subjects(scope = DERIV_WORKFLOW)
    for sub in subs:
        if sub in already_done:
            continue
        else:
            main(sub, layout)
