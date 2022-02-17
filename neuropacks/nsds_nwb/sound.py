import numpy as np

# this module is separated so that soundsig is not required beyond TIMIT/DMR
from soundsig.sound import spectrogram


def make_spectrogram(sound_waveform, raw_rate, spec_sample_rate=1000,
                     freq_spacing=50, min_freq=0, max_freq=10000,
                     dbnoise=50):
    # adapted from soundsig.sound.BioSound.spectroCalc()
    # Calculates the spectrogram in dB
    t, f, spec, _ = spectrogram(sound_waveform, raw_rate,
                                spec_sample_rate=spec_sample_rate,
                                freq_spacing=freq_spacing,
                                min_freq=min_freq, max_freq=max_freq,
                                cmplx=True)
    spectro = 20 * np.log10(np.abs(spec))

    # adapted from how soundsig.sound.BioSound.plot() handles spectrogram
    maxB = spectro.max()
    minB = maxB - dbnoise
    spectro[spectro < minB] = minB

    return t, f, spectro
