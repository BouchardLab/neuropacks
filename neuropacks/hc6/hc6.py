import numpy as np
import os

from scipy.io import loadmat


class HC6():
    def __init__(self, directory):
        self.directory = directory
        self.base = os.path.basename(os.path.normpath(self.directory)).lower()
        self.__initialize_class()

    def __initialize_class(self):
        # first look at cell info
        cell_info_path = os.path.join(self.directory,
                                      self.base + 'cellinfo.mat')
        cell_info = loadmat(cell_info_path, struct_as_record=False)['cellinfo']

        # discover the valid days
        self.n_days = cell_info[0].size
        self.n_epochs = np.array([day.size for day in cell_info[0]])
        self.valid_days = np.argwhere(self.n_epochs > 0).ravel()
        self.valid_day_labels = self.valid_days + 1

        # number of tetrodes
        self.n_tetrodes = np.zeros(self.n_days, dtype='int')
        for day in self.valid_days:
            self.n_tetrodes[day] = cell_info[0, day][0, 0].size

        # number of units
        self.n_units_per_tetrode = {}
        self.n_units = {}
        for day in range(self.n_days):
            self.n_units_per_tetrode[day] = {}
            self.n_units[day] = {}
            for epoch in range(self.n_epochs[day]):
                self.n_units_per_tetrode[day][epoch] = {}
                self.n_units[day][epoch] = 0
                for tetrode in range(self.n_tetrodes[day]):
                    n_units = cell_info[0, day][0, epoch][0, tetrode].size
                    self.n_units_per_tetrode[day][epoch][tetrode] = n_units
                    self.n_units[day][epoch] += n_units

        # get positions
        self.positions = {}
        for day in self.valid_days:
            self.positions[day] = {}
            position_path = os.path.join(self.directory,
                                         self.base + 'pos%02d' % (day + 1)
                                         + '.mat')
            positions = loadmat(position_path, struct_as_record=False)
            for epoch in range(self.n_epochs[day]):
                data = positions['pos'][0, day][0, epoch][0, 0].data
                self.positions[day][epoch] = {}
                self.positions[day][epoch]['time'] = data[:, 0]
                self.positions[day][epoch]['x'] = data[:, 1]
                self.positions[day][epoch]['y'] = data[:, 2]

        # get spike times
        self.spike_times = {}
        for day in self.valid_days:
            self.spike_times[day] = {}
            spikes_path = os.path.join(self.directory,
                                       self.base + 'spikes%02d' % (day + 1)
                                       + '.mat')
            spike_times = loadmat(spikes_path, struct_as_record=False)
            for epoch in range(self.n_epochs[day]):
                self.spike_times[day][epoch] = {}
                for tetrode in range(self.n_tetrodes[day]):
                    for unit in range(self.n_units_per_tetrode[day][epoch][tetrode]):
                        self.spike_times[day][epoch][unit]

