from parselmouth.praat import call as praat
from parselmouth import Sound
import numpy as np

MIN_DIP = 2.        # minimum peak-to-valley intensity difference for syllables
MIN_PAUSE = 0.3     # minimum pause duration
MIN_PITCH = 40.     # minimum_pitch for intensity estimation
MAX_PITCH = 300.    # maximum pitch for determining whether voicing

def articulation_rate(raw, tmin = 0., tmax = None, channel = 'audio'):
    '''
    finds number of syllable nuclei and the phonation time, which gives
    you the articulation rate. (But we return them seperately to give you
    what you need for a Poisson regression.)

    works directly on an mne.Raw (or single Epoch) object, which should
    contain only one trial.

    based on code from DOI 10.17605/OSF.IO/6DWR3
    or https://github.com/drfeinberg/PraatScripts/blob/master/syllable_nuclei.py
    which implements:

    De Jong, N. H., & Wempe, T. (2009). Praat script to detect syllable nuclei
    and measure speech rate automatically.
    Behavior research methods, 41(2), 385-390.

    Updates to the above code (since the 2009 publication) removed the median
    intensity threshold in favor of a  default threshold of 25 dB lower than the
    maximum intensity, but that can perform poorly in our case due to background
    noise so we return to the original median theshold method. The addition of
    background (usually pink) noise is common in delayed auditory feedback
    experiments to mask bone-conducted acoustics, so this will generally be more
    appropriate if estimating articulation rate based on the recorded feedback
    audio. Both methods should perform quite well if estimating from the
    electroglottograph, at least after applying a 40 Hz-ish highpass
    to remove non-phonotory (e.g. swallowing) artifacts.
    '''

    # pull audio out of MNE object
    audio = raw.copy().crop(tmin, tmax).get_data(picks = [channel])
    audio = np.squeeze(audio)
    if len(audio.shape) != 1:
        raise ValueError('input MNE Object must contain only one trial')
    snd = Sound(audio, raw.info['sfreq'])

    intensity = snd.to_intensity(minimum_pitch = 60.)
    min_intensity = praat(intensity, "Get minimum", 0, 0, "Parabolic")
    max_intensity = praat(intensity, "Get maximum", 0, 0, "Parabolic")

    # estimate intensity threshold
    median_intensity = praat(intensity, "Get quantile", 0, 0, 0.5)
    threshold = median_intensity
    threshold = threshold if threshold > min_intensity else min_intensity
    silence_db = threshold - max_intensity # dB relative to maximum

    # find pauses (silences)
    textgrid = praat(intensity, "To TextGrid (silences)", silence_db,
                          MIN_PAUSE, 0.1, "silent", "sounding")
    silence_tier = praat(textgrid, "Extract tier", 1)
    silence_table = praat(silence_tier, "Down to TableOfReal", "sounding")
    n_pauses = praat(silence_table, "Get number of rows")
    # then compute speaking time as total minus silent time
    speaking_time = 0
    for pause in range(n_pauses):
        begin = praat(silence_table, "Get value", pause + 1, 1)
        end = praat(silence_table, "Get value", pause + 1, 2)
        dur = end - begin
        speaking_time += dur

    # estimate preliminary peak positions from intensity matrix
    intensity_matrix = praat(intensity, "Down to Matrix")
    sound_from_intensity_matrix = praat(intensity_matrix, "To Sound (slice)", 1)
    point_process = praat(sound_from_intensity_matrix,
                     "To PointProcess (extrema)", "Left", "yes", "no", "Sinc70")
    n_peaks = praat(point_process, "Get number of points")
    t = [praat(point_process, "Get time from index", i + 1)
                            for i in range(n_peaks)]

    # and filter for above-threshold peaks
    vals = [praat(sound_from_intensity_matrix,
                "Get value at time", t[i], "Cubic") for i in range(n_peaks)]
    peaks = [val for val in vals if val > threshold]
    t_peaks = [t[i] for i, val in enumerate(vals) if val > threshold]
    n_peaks = len(peaks)

    # then filter for peaks which meet peak-to-valley (dip) threshold
    valleys = [praat(intensity, "Get minimum", t_peaks[i], t_peaks[i + 1], "None")
                       for i in range(n_peaks - 1)]
    valleys = np.array(valleys)
    peaks, t_peaks = np.array(peaks[:-1]), np.array(t_peaks[:-1])
    valid = (peaks - valleys) > MIN_DIP
    peaks, t_peaks = peaks[valid], t_peaks[valid]

    # and finally filter for peaks for which there is clear voicing
    intervals = [praat(textgrid, "Get interval at time", 1, t) for t in t_peaks]
    labels = [praat(textgrid, "Get label of interval", 1, interval)
                    for interval in intervals]
    pitch_tier = snd.to_pitch_ac(0.02, 40, 4, False, 0.03, 0.25, 0.01, 0.35, 0.25, MAX_PITCH)
    pitches = [pitch_tier.get_value_at_time(t) for t in t_peaks]
    voiced_idxs = [i for i, p in enumerate(pitches) if (not np.isnan(p)) and p > 0]
    voiced_idxs = [idx for idx in voiced_idxs if labels[idx] == 'sounding']
    n_peaks = len(voiced_idxs)

    #articulation_rate = n_peaks / speaking_time
    return n_peaks, speaking_time

def get_articulation_rates(raw, events, channel = 'audio'):

    n_syllables = []
    speech_durs = []

    for ev in range(events.shape[0]):
        start = events.onset[ev]
        end = events.onset[ev] + events.duration[ev]
        n, dur = articulation_rate(raw, start, end, channel)
        n_syllables.append(n)
        speech_durs.append(dur)

    n_syllables = np.array(n_syllables)
    speech_durs = np.array(speech_durs)
    return n_syllables, speech_durs
