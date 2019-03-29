import numpy as np
import os

from scipy.io import loadmat


class HC6():
    def __init__(self, directory):
        self.directory = directory
        self.base = os.path.basename(os.path.normpath(self.directory)).lower()
        self.__initialize_class()

    def __initialize_class(self):
        """Initializes the datasets in the provided directory."""
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

        # data dictionaries
        self.n_units = {}
        self.n_units_per_tetrode = {}
        self.positions = {}
        self.spike_times = {}

        # iterate over the days
        for day in range(self.n_days):
            # initialize top level entry of dictionaries
            self.n_units[day] = {}
            self.n_units_per_tetrode[day] = {}
            self.positions[day] = {}
            self.spike_times[day] = {}

            if day in self.valid_days:
                # obtain data for position
                position_path = os.path.join(
                    self.directory,
                    self.base + 'pos%02d' % (day + 1) + '.mat'
                )
                positions = loadmat(
                    position_path,
                    struct_as_record=False
                )['pos'][0, day]

                # obtain data for spikes
                spikes_path = os.path.join(
                    self.directory,
                    self.base + 'spikes%02d' % (day + 1) + '.mat'
                )
                spikes = loadmat(
                    spikes_path,
                    struct_as_record=False
                )['spikes'][0, day]

            # iterate over epochs in day
            for epoch in range(self.n_epochs[day]):
                self.n_units[day][epoch] = 0
                self.n_units_per_tetrode[day][epoch] = {}
                self.spike_times[day][epoch] = {}

                # grab positions
                xy = positions[0, epoch][0, 0].data
                positions_epoch = {}
                positions_epoch['time'] = xy[:, 0]
                positions_epoch['x'] = xy[:, 1]
                positions_epoch['y'] = xy[:, 2]
                self.positions[day][epoch] = positions_epoch

                # we will be tetrode agnostic when we store the spike times of
                # the units
                unit_idx = 0
                # iterate over tetrodes
                for tetrode in range(self.n_tetrodes[day]):
                    n_units = spikes[0, epoch][0, tetrode].size
                    self.n_units_per_tetrode[day][epoch][tetrode] = n_units
                    self.n_units[day][epoch] += n_units

                    # iterate over units in the tetrode
                    for unit in range(n_units):
                        unit_data = spikes[0, epoch][0, tetrode][0, unit]
                        if unit_data.size != 0:
                            spike_times = unit_data[0, 0].data
                            if spike_times.size > 0:
                                self.spike_times[day][epoch][unit_idx] = \
                                    spike_times[:, 0]
                            else:
                                self.spike_times[day][epoch][unit_idx] = \
                                    np.array([])
                        else:
                            self.spike_times[day][epoch][unit_idx] = \
                                np.array([])
                        unit_idx += 1

    def get_tuning_matrix(self, day, epoch, n_gaussians):
        positions = self.positions[day][epoch]
        xs = positions['x']
        ys = positions['y']

        x_std = np.std(xs)
        y_std = np.std(ys)
        x_means = x_std * np.linspace(-2, 2, n_gaussians) + xs.mean()
        y_means = y_std * np.linspace(-2, 2, n_gaussian) + ys.mean()

        tuning_std_x = x_std / 2
        tuning_std_y = y_std / 2

        X = np.zeros(())
        for idx, (x, y) in enumerate(zip(xs, ys)):

