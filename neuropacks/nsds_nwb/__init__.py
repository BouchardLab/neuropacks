from pynwb import NWBHDF5IO

from .nsds_nwb import NSDSNWBAudio, detect_stim_for_session
from .continuous import TIMIT, DynamicMovingRipples
from .discrete import Tone, WhiteNoise


def get_nsds_nwb_neuropack(nwb_path, stim_name=None):
    '''Wrapper function to select and construct the correct neuropack class
    according to the stimulus type used in the block.

    Parameters
    ----------
    nwb_path : str or pathlike
        Path to the NWB file.
    stim_name : str
        Specifies the stimulus name, to determine which neuropack to use.
        Standard options are ('dmr', 'timit', 'tone*', 'wn*')
        but alternative names like 'Ripples' or 'White noise' are allowed; see
        full list in `nsds_lab_to_nwb.metadata.resources.list_of_stimuli.yaml`.
        If None, stimulus name is auto-detected from the NWB file.

    Returns
    -------
    Returns a neuropack instance (of an appropriate subclass of NSDSNWBAudio),
    depending on the stimulus type.
    '''
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
