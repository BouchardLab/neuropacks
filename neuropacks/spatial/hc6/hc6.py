import numpy as np
import os

from scipy.io import loadmat


class HC6():
    def __init__(self, directory, base=None):
        """Processes and provides response matrices for the hc6 hippocampus
        dataset obtained from the Frank lab.

        Parameters
        ----------
        directory : string
            The path to the top level folder containing the data.

        Attributes
        ----------
        base : string

        n_days : int

        valid_days : ndarray

        valid_days_labels : ndarray

        n_tetrodes : ndarray

        n_units : nested dict

        n_units_per_tetrode : nested dict

        positions : nested dict

        spike_times : nested dict
        """
        self.directory = directory

        if base is None:
            base = os.path.basename(os.path.normpath(self.directory)).lower()
        self.base = base

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

    def get_positions(self, day, epoch, return_time=False):
        """Obtain the animal's position by time for a given experiment.

        Parameters
        ----------
        day : int
            The index of the day.

        epoch : int
            The index of the epoch.

        return_time : bool
            If True, the timestamps will be returned.

        Returns
        -------
        xs : ndarray
            The x positions.

        ys : ndarray
            The y positions.

        times : ndarray
            The timestamps for each position; optional.
        """
        if day not in self.valid_days:
            raise ValueError('Day does not have any recordings.')

        positions = self.positions[day][epoch]
        times = positions['time']
        xs = positions['x']
        ys = positions['y']

        if return_time:
            return xs, ys, times
        else:
            return xs, ys

    def get_spike_times(self, day, epoch, unit):
        """Obtain spike times by day, epoch and unit.

        Parameters
        ----------
        day : int
            The index of the day.

        epoch : int
            The index of the epoch.

        unit : int
            The index of the unit.

        Returns
        -------
        spike_times : ndarray
            The timestamps for each spike; optional.
        """
        if day not in self.valid_days:
            raise ValueError('Day does not have any recordings.')

        max_unit_idx = self.n_units[day][epoch]
        if (unit >= max_unit_idx) or (unit < 0):
            raise ValueError('Unit index does not exist.')

        spike_times = self.spike_times[day][epoch][unit]

        return spike_times

    def get_binned_positions(self, day, epoch, bin_width=0.5):
        """Bin the animal's positions according to a bin width.

        Parameters
        ----------
        day : int
            The index of the day.

        epoch : int
            The index of the epoch.

        bin_width : float
            The width of the bin, in seconds.

        Returns
        -------
        x_binned : ndarray
            The mean x position in each bin.

        y_binned : ndarray
            The mean y position in each bin.
        """
        xs, ys, times = self.get_positions(day, epoch, return_time=True)

        # calculate number of bins
        n_bins = int(np.ceil(
            (times[-1] - times[0])/bin_width
        ))
        bins = np.arange(n_bins + 1) * bin_width + times[0]
        bin_indices = np.digitize(times, bins=bins) - 1

        x_binned = np.zeros(n_bins)
        y_binned = np.zeros(n_bins)

        for bin_idx in range(n_bins):
            indices = np.argwhere(bin_indices == bin_idx).ravel()
            x_binned[bin_idx] = np.mean(xs[indices])
            y_binned[bin_idx] = np.mean(ys[indices])

        return x_binned, y_binned

    def get_design_matrix(self, day, epoch, bin_width=0.5, n_gaussians=5):
        """Parameterize the positions using gaussian basis functions.

        Parameters
        ----------
        day : int
            The index of the day.

        epoch : int
            The index of the epoch.

        bin_width : float
            The width of the bin, in seconds.

        n_gaussians : int
            The number of gaussians to use in one dimension (the total number
            of basis functions will be n_gaussians**2).

        Returns
        -------
        X : ndarray, shape (n_samples, n_gaussians**2)
            The design matrix.
        """
        xs, ys, times = self.get_positions(day, epoch, return_time=True)

        # gaussian basis functions
        x_std = np.std(xs)
        y_std = np.std(ys)

        # calculate means of gaussian bfs
        x_bf_means = x_std * np.linspace(-2, 2, n_gaussians) + xs.mean()
        y_bf_means = y_std * np.linspace(-2, 2, n_gaussians) + ys.mean()
        means = np.transpose([
            np.tile(x_bf_means, n_gaussians),
            np.repeat(y_bf_means, n_gaussians)
        ])

        # calculate stds of gaussian bfs
        x_bf_std = x_std / 2
        y_bf_std = y_std / 2
        covar = np.array([[x_bf_std**2, 0], [0, y_bf_std**2]])
        inv_covar = np.linalg.inv(covar)
        norm = 1. / np.sqrt((2 * np.pi)**2 * np.linalg.det(covar))

        x_binned, y_binned = self.get_binned_positions(
            day=day,
            epoch=epoch,
            bin_width=bin_width
        )
        xy_binned = np.transpose([x_binned, y_binned])
        n_samples = x_binned.size

        X = np.zeros((n_samples, n_gaussians**2))

        for sample_idx, xy in enumerate(xy_binned):
            for gaussian_idx, mean in enumerate(means):
                arg = xy - mean
                X[sample_idx, gaussian_idx] = norm * np.exp(
                    -0.5 * np.dot(arg, np.dot(inv_covar, arg))
                )

        return X, (means, covar)

    def get_response_matrix(
        self, day, epoch, bin_width, transform='square_root', clean=False
    ):
        """Create the response matrix by binning the spike times.

        Parameters
        ----------
        bin_width : float
            The width of the bins, in seconds.

        transform : string
            The transform to apply to the spike counts. Available option is
            'square_root'. If None, no transform is applied.

        Returns
        -------
        Y : nd-array, shape (n_trials, n_neurons)
            Response matrix containing (transformed) spike counts of each
            neuron.
        """
        # grab times
        _, _, times = self.get_positions(day, epoch, return_time=True)

        # number of units
        spike_times = self.spike_times[day][epoch]
        n_units = len(spike_times)

        # calculate number of bins
        n_bins = int(np.ceil(
            (times[-1] - times[0])/bin_width
        ))
        bins = np.arange(n_bins + 1) * bin_width + times[0]

        Y = np.zeros((n_bins, n_units))
        silent_neurons = np.array([])
        # iterate over spike time arrays in corresponding dictionary
        for idx, (key, spikes) in enumerate(spike_times.items()):
            # bin the spike times
            binned_spike_counts = np.histogram(spikes, bins=bins)[0]

            # apply a transform
            if transform is None:
                Y[:, idx] = binned_spike_counts
            elif transform == 'square_root':
                Y[:, idx] = np.sqrt(binned_spike_counts)
            else:
                raise ValueError('Transform %s is not valid.' % transform)

            if spikes.size == 0:
                silent_neurons = np.append(silent_neurons, idx)

        if clean:
            Y = np.delete(Y, silent_neurons, axis=1)

        return Y
