import numpy as np

from process_nwb.resample import resample

from neuropacks.nsds_nwb.nsds_nwb import NSDSNWBAudio
from neuropacks.nsds_nwb.sound import make_spectrogram
from neuropacks.nsds_nwb.utils import normalize_neural


class TIMIT(NSDSNWBAudio):
    def __init__(self, nwb_path):
        super().__init__(nwb_path)

    def get_design_matrix(self, sample_rate=100., encoding='spectro', **kwargs):
        """Create the trial design matrix.

        **** TENTATIVE FORMATS - work in progress ****

        Parameters
        ----------
        sample_rate : float
            Frequency in Hz to downsample the stimuli to.
        encoding : str
            How the design matrix should be encoded.
            Options are ['wav', 'envelope', 'spectro'].

        Returns
        -------
        design : list
            List where each item is for a stimulus trial.
            Each item includes the stimulus design matrix and optionally other info.
        """
        idxs = (self.intervals['sb'] == 's')
        self.n_trials = len(idxs)   # stimulus trials only!

        if 'wav' in encoding:
            stim_list = self._get_trialized_stim(type='wav')
            # design = stim_list
            # downsample to sample_rate?? not very informative
            design = []
            for stim in stim_list:
                resampled = resample(stim.data, sample_rate, stim.rate,
                                     real=True, axis=0)
                design.append(resampled)

        elif 'env' in encoding:
            design = self._get_trialized_stim(type='env')
            # raise NotImplementedError('TODO')

        elif 'spectro' in encoding:
            stim_list = self._get_trialized_stim(type='wav')
            design = []
            for stim in stim_list:
                t, f, spectro = make_spectrogram(stim.data, stim.rate,
                                                 spec_sample_rate=sample_rate,
                                                 **kwargs)
                t_shifted = t + stim.starting_time
                design.append((t_shifted, f, spectro))
        else:
            raise ValueError(f"`encoding` must be one of ['wav', 'spectro']; was {encoding}")
        return design

    def _get_trialized_spectrograms(self):
        stim_ns = self._get_stimulus_waveform()

        stim_wav_list = []
        for ii, row in self.intervals.iterrows():
            if row['sb'] != 's':
                continue

            if in_memory:
                idx = slice_interval(row['start_time'], row['stop_time'],
                                     rate=stim_ns.rate,
                                     t_offset=stim_ns.starting_time)
                stim_sliced = stim_ns.data[idx]
            else:
                stim_sliced = self._get_stimulus_waveform(
                    start_time=row['start_time'],
                    stop_time=row['stop_time']).data
            stim_wav_list.append(stim_sliced)
        return stim_wav_list

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
