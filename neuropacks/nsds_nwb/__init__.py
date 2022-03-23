from pynwb import NWBHDF5IO

from .nsds_nwb import NSDSNWBAudio, detect_stim_for_session
from .continuous import TIMIT, DynamicMovingRipples
from .discrete import Tone, WhiteNoise


def get_nsds_nwb_neuropack(nwb_path, stim_name=None):
    if stim_name is None:
        with NWBHDF5IO(nwb_path, 'r') as io:
            nwb = io.read()
            stim_name, _ = detect_stim_for_session(nwb.session_description)

    if stim_name is None:
        raise ValueError('Stimulus name could not be automatically detected. '
                         'Please provide a stim_name.')

    if stim_name == 'dmr':
        return DynamicMovingRipples(nwb_path)
    if stim_name == 'timit':
        return TIMIT(nwb_path)
    if 'tone' in stim_name:
        return Tone(nwb_path)
    if 'wn' in stim_name:
        return WhiteNoise(nwb_path)

    raise ValueError(f'Unknown stimulus name {stim_name}.'
                     'Recognized names are [dmr, timit, tone*, wn*].')
