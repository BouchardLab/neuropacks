import numpy as np

from neuropacks.nsds_nwb.nsds_nwb import NSDSNWBAudio
from neuropacks.nsds_nwb.utils import normalize_neural


class TIMIT(NSDSNWBAudio):
    def __init__(self, nwb_path):
        super().__init__(nwb_path)

    def get_design_matrix(self):
        raise NotImplementedError

    def get_response_matrix(self, neural_data='ecog', in_memory=True):
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

        responses_list, baselines_list = self._get_trialized_responses(
            neural_data, in_memory=in_memory)

        baseline = np.concatenate(baselines_list, axis=0)   # (timepoints, channels)
        response = []
        for res in responses_list:
            # add a new axis 0 for normalize_neural
            res, _, _ = normalize_neural(res[np.newaxis], baseline)
            res_trial = np.transpose(res[0].mean(axis=-1), (1, 0))  # (channels, timepoints)
            response.append(res_trial)
        return response