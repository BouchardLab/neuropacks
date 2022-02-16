import numpy as np
import pandas as pd
import sklearn.preprocessing as skpre

from neuropacks.nsds_nwb.nsds_nwb import NSDSNWBAudio


class Tone(NSDSNWBAudio):
    def __init__(self, nwb_path):
        super().__init__(nwb_path)
        self.intervals['frq'] = self.intervals['frq'].astype(float)
        self.intervals['amp'] = self.intervals['amp'].astype(float)
        self.unique_frequencies = np.unique(self.intervals['frq'])
        self.unique_amplitudes = np.unique(self.intervals['amp'])

    def get_design_matrix(self, encoding='label'):
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

    def get_response_matrix(self):
        raise NotImplementedError
