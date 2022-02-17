import numpy as np
import pandas as pd
import sklearn.preprocessing as skpre

from pynwb import NWBHDF5IO

from neuropacks.nsds_nwb.nsds_nwb import NSDSNWBAudio
from neuropacks.nsds_nwb.utils import normalize_neural


class Tone(NSDSNWBAudio):
    def __init__(self, nwb_path):
        super().__init__(nwb_path)
        self.intervals['frq'] = self.intervals['frq'].astype(float)
        self.intervals['amp'] = self.intervals['amp'].astype(float)
        self.unique_frequencies = np.unique(self.intervals['frq'])
        self.unique_amplitudes = np.unique(self.intervals['amp'])

    def get_design_matrix(self, encoding='label'):
        """Create the trial design matrix.

        Parameters
        ----------
        encoding : str
            How the design matrix should be encoded.
            Options are ['label', 'value', 'onehot']

        Returns
        -------
        design : dataframe
            Design matrix dataframe. Each row is a trial and each column is a stimulus parameter.
        """
        idxs = self.intervals['sb'] == 's'
        self.n_trials = len(idxs)
        if encoding in ['label', 'onehot']:
            frq_enc = skpre.LabelEncoder()
            frq_enc.fit(self.unique_frequencies)
            amp_enc = skpre.LabelEncoder()
            amp_enc.fit(self.unique_amplitudes)
            design = pd.DataFrame({'frequency': frq_enc.transform(self.intervals['frq'].loc[idxs]),
                                   'amplitude': amp_enc.transform(self.intervals['amp'].loc[idxs])})
            if encoding == 'onehot':
                frq_enc1 = skpre.OneHotEncoder()
                frq_enc1.fit(frq_enc.transform(self.unique_frequencies)[:, np.newaxis])
                amp_enc1 = skpre.OneHotEncoder()
                amp_enc1.fit(amp_enc.transform(self.unique_amplitudes)[:, np.newaxis])
                frqs = frq_enc1.transform(design['frequency'][:, np.newaxis]).toarray().astype(int)
                amps = amp_enc1.transform(design['amplitude'][:, np.newaxis]).toarray().astype(int)
                design = pd.DataFrame({'frequency': frqs.tolist(),
                                       'amplitude': amps.tolist()})
        elif encoding == 'value':
            design = pd.DataFrame({'frequency': self.intervals['frq'].loc[idxs],
                                   'amplitude': self.intervals['amp'].loc[idxs]})
        else:
            raise ValueError(f"`encoding` must be one of ['label', 'value', 'onehot'] was {encoding}")
        return design

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
        """
        if neural_data not in ['ecog', 'poly']:
            raise ValueError(f"`neural_data` should be one of ['ecog', 'poly'], got {neural_data}")

        response, baseline = self._get_trialized_responses(
            neural_data, in_memory=in_memory)

        baseline = np.concatenate(baseline, axis=0)
        response = np.stack(response)
        response, mean, std = normalize_neural(response, baseline)
        return np.transpose(response.mean(axis=-1), (0, 2, 1))

    def _get_trialized_responses(self, neural_data, in_memory=True):
        ''' overrides NSDSNWBAudio method; essentially the same. should confirm
        '''
        data = []
        with NWBHDF5IO(self.nwb_path, 'r') as io:
            nwb = io.read()
            di = nwb.processing['preprocessing'].data_interfaces
            for n in di.keys():
                if neural_data in n.lower():
                    if in_memory:
                        data.append(di[n].data[:])
                    else:
                        data.append(di[n].data)
                    rate = di[n].rate
                    starting_time = di[n].starting_time
            if len(data) == 1:
                idxs = self.electrode_df['group_name'] == 'ECoG'
                good_electrodes = ~self.electrode_df['bad'].loc[idxs].values
                data = data[0]
            else:
                raise ValueError(f'Multiple {neural_data} entries found.')
            baseline = []
            response = []
            for ii, row in self.intervals.iterrows():
                start = row['start_time']
                stop = row['stop_time']
                starti = int((start - starting_time) * rate)
                stopi = int((stop - starting_time) * rate)
                if row['sb'] == 'b':
                    baseline.append(data[starti:stopi][:, good_electrodes])
                if row['sb'] == 's':
                    response.append(data[starti:stopi][:, good_electrodes])
        return response, baseline
