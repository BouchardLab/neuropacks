import h5py
import numpy as np


class BG():
    def __init__(self, data_path):
        self.data_path = data_path
        data = h5py.File(data_path, 'r')

        if 'GoodUnits' in data:
            self.good_units = data['GoodUnits'][:].ravel()
        else:
            self.good_units = None

        trials = data['Trials']
        n_trials = trials['time'].shape[0]

        for trial_idx in range(n_trials):




class Trial():
    def __init__(self, )