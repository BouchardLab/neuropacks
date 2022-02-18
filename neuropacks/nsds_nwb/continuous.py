import numpy as np

from neuropacks.nsds_nwb.nsds_nwb import NSDSNWBAudio
from neuropacks.nsds_nwb.utils import normalize_neural


class ContinuousStimuli(NSDSNWBAudio):
    '''Neuropack for NSDS auditory experiment with continuous stimulus type.

    The "continuous" stimulus type has two properties:
        - Stimulus does not have simple parameterization (like WN or Tone).
        - Trial lengths could vary from trial to trial.

    TIMIT and DynamicMovingRipples (DMR) blocks should use this class.
    '''
    def __init__(self, nwb_path):
        super().__init__(nwb_path)

    def get_design_matrix(self):
        """Create the trial design "matrix".

        For continuous stimuli, simply returns the stimulus waveforms
        (wrapped into a SimpleNamespace object) from each stimulus trial.
        For timit998, for example, there should be 998 trials.

        We could add more options for stimulus encoding (not just waveforms)
        as appropriate for individual stimulus types.

        Returns
        -------
        design : list
            List of SimpleNamespace objects. Each namespace item includes the
            stimulus waveform data from a stimulus trial, along with other metadata.
        """
        design = self.get_trialized_stim(type='waveform')
        self.n_trials = len(design)   # stimulus trials only!
        return design

    def get_response_matrix(self, neural_data='ecog', in_memory=True, band_idx=None):
        """Create the neural response matrix.

        Parameters
        ----------
        neural_data : str
            Which neural dataset to load.
            Options are ['ecog', 'poly']
        in_memory : bool
            Whether to load the entire dataset into memory intermediately before trializing it. This
            is considerably faster if the whole dataset fits into memory.

        Returns:
        --------
        response : list of ndarrays
            List of response matrices, each response matrix from a stimulus trial.
            Each item is an ndarray with shape (n_channels, n_timepoints)
        """
        if neural_data not in ['ecog', 'poly']:
            raise ValueError(f"`neural_data` should be one of ['ecog', 'poly'], got {neural_data}")

        responses_list, baselines_list = self.get_trialized_responses(
            neural_data, in_memory=in_memory)

        if band_idx is None:
            band_idx = slice(None)

        baseline = np.concatenate(baselines_list, axis=0)   # (timepoints, channels)
        response = []
        for res in responses_list:
            # add a new axis 0 for normalize_neural
            res, _, _ = normalize_neural(res[np.newaxis], baseline)

            # average over select bands
            res_bnd = np.mean(res[0][..., band_idx], axis=-1)

            res_trial = np.transpose(res_bnd, (1, 0))  # (channels, timepoints)
            response.append(res_trial)
        return response


class TIMIT(ContinuousStimuli):
    def __init__(self, nwb_path):
        super().__init__(nwb_path)


class DynamicMovingRipples(ContinuousStimuli):
    def __init__(self, nwb_path):
        super().__init__(nwb_path)
