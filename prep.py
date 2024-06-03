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
from bids import BIDSLayout
# speech stuff
from parselmouth import Sound

BIDS_ROOT = 'bids_dataset'
DERIV_WORKFLOW = 'PREP'
DERIV_ROOT = op.join(BIDS_ROOT, 'derivatives')
TASK = 'daf'

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

    ## find good filter settings for EGG
    audio = raw.copy().pick(['egg'])
    audio = audio.load_data()
    # get min and max pitch across all trials
    min_pitch = np.inf
    max_pitch = -np.inf
    for i in range(evs.shape[0]):
        tmin, tmax = evs.onset.iloc[i], evs.onset.iloc[i] + evs.duration.iloc[i]
        aud, t = audio.get_data(tmin = tmin, tmax = tmax, return_times = True)
        aud = np.squeeze(aud)
        snd = Sound(aud, sampling_frequency = audio.info['sfreq'])
        pitch = snd.to_pitch().selected_array['frequency']
        min_pitch = np.min([pitch[pitch > 0.].min(), min_pitch])
        max_pitch = np.max([pitch[pitch > 0.].max(), max_pitch])

    # filter EGG before downsampling for PREP
    raw.load_data()
    raw.filter(min_pitch - 10., max_pitch + 10., picks = ['egg'])
    # and audio
    raw.filter(20., 5000/2, picks = ['audio'])

    # re-reference eye electrodes to become bipolar EOG
    def ref(dat):
        dat[0,:] = (dat[0,:] - dat[1,:])
        return dat
    raw = raw.apply_function(ref, picks = ['leog', 'Fp2'], channel_wise = False)
    raw = raw.apply_function(ref, picks = ['reog', 'Fp1'], channel_wise = False)
    raw = raw.set_channel_types({'leog': 'eog', 'reog': 'eog'})

    # run PREP pipeline (notch, exclude bad chans, and re-reference)
    raw = raw.resample(5000.) # downsample for PREP as a workaround for
    lf = raw.info['line_freq'] # pyprep Issue #109 on GitHub:
    prep_params = {            # https://github.com/sappelhoff/pyprep/issues/109
        "ref_chs": "eeg",
        "reref_chs": "eeg",
        "line_freqs": np.arange(lf, 200., lf)
    }
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
    raw = raw.apply_function(ref, picks = ['FT10', 'TP10'], channel_wise = False)
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
