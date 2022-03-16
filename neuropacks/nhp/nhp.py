import h5py
import numpy as np
import pandas as pd


class NHP:
    def __init__(self, data_path):
        """Processes and provides response matrices for the non-human primate
        motor cortex dataset obtained from the Sabes lab.

        Parameters
        ----------
        data_path : string
            The path to the M1 dataset.

        Attributes
        ----------
        data_path : string
            The path to the NHP dataset.

        chan_names : list
            List of channel names, formatted as '{region} {channel_idx}',
            for example 'M1 001'.

        timestamps : nd-array, shape (n_timestamps)
            The timestamps, in seconds, for the session.

        cursor_pose : nd-array, shape (2, n_timestamps)
            The cursor position in x,y coordinates (mm) at each timestamp.

        target_pos : nd-array, shape (2, n_timestamps)
            The target position in x,y coordinates (mm) at each timestamp.

        finger_pos : nd-array, shape (6, n_timestamps)
            The position of the working fingertip in Cartesian coordinates
            (z, -x, -y), as reported by the hand tracker in cm.

        n_sorted_units : int
            The number of sorted units per channel. This number excludes the
            unsorted units.

        spike_times : dict
            Merged dictionary {'M1': M1_spike_times, 'S1': S1_spike_times}.

        M1_spike_times : dict
            A dictionary where each key denotes a channel, unit combination in
            M1 and value is an nd-array of spike times, in seconds, during the
            session.

        S1_spike_times : dict
            A dictionary where each key denotes a channel, unit combination in
            S1 and value is an nd-array of spike times, in seconds, during the
            session.
        """
        # open up .mat file
        self.data_path = data_path
        data = h5py.File(data_path, 'r')
        self.chan_names = self._read_chan_names(data)
        self.timestamps = data['t'][0, :]
        self.cursor_pos = data['cursor_pos'][:]
        self.target_pos = data['target_pos'][:]
        self.finger_pos = data['finger_pos'][:]

        self.spike_times, self.n_sorted_units = self._read_spike_times(data)
        data.close()

        # keep legacy attributes
        self.M1_spike_times = self.spike_times['M1']
        self.S1_spike_times = self.spike_times['S1']

        self.trials, self.target_grid = self._parse_trials()

    def get_binned_positions(self, bin_width=0.5):
        """Bin the mouse positions according to a bin width.

        Parameters
        ----------
        bin_width : float
            The width of the bin, in seconds.

        Returns
        -------
        x_binned : ndarray
            The mean x position in each bin.

        y_binned : ndarray
            The mean y position in each bin.
        """
        return self.bin_timeseries(self.cursor_pos, bin_width=bin_width)

    def bin_timeseries(self, data_ts, bin_width=0.5):
        bins, bin_indices, n_bins = self._bin_times(bin_width)

        data_binned = np.zeros((n_bins, 2))
        for bin_idx in range(n_bins):
            indices = np.argwhere(bin_indices == bin_idx).ravel()
            data_binned[bin_idx] = np.mean(data_ts[:, indices], axis=1).T
        return data_binned

    def get_response_matrix(self, bin_width, region='M1', transform='sqrt'):
        """Create the response matrix by binning the spike times.

        Parameters
        ----------
        bin_width : float
            The width of the bins, in seconds.

        region : string
            The region to create a response matrix for ('M1' or 'S1').

        transform : string
            The transform to apply to the spike counts.
            Available option is 'square_root' (abbrev. 'sqrt').
            If None, no transform is applied.

        Returns
        -------
        Y : nd-array, shape (n_trials, n_neurons)
            Response matrix containing (transformed) spike counts of each
            neuron.
        """
        try:
            spike_times = self.spike_times[region]
        except KeyError:
            raise ValueError(f'Region {region} is neither M1 nor S1.')

        bins, _, n_bins = self._bin_times(bin_width)

        Y = np.zeros((n_bins, len(spike_times)))

        # iterate over spike time arrays in corresponding dictionary
        for idx, (key, spikes) in enumerate(spike_times.items()):
            # bin the spike times
            binned_spike_counts = np.histogram(spikes, bins=bins)[0]

            # apply a transform
            if transform is None:
                Y[:, idx] = binned_spike_counts
            elif transform in ('sqrt', 'square_root'):
                Y[:, idx] = np.sqrt(binned_spike_counts)
            else:
                raise ValueError(f'Transform {transform} is not valid.')

        return Y

    def get_binned_times(self, bin_width):
        bins, _, _ = self._bin_times(bin_width)
        times_bin_center = (bins[1:] + bins[:-1]) / 2
        return times_bin_center

    def _bin_times(self, bin_width):
        # calculate number of bins
        n_bins = int(np.ceil(
            (self.timestamps[-1] - self.timestamps[0])/bin_width
        ))

        bins = np.arange(n_bins + 1) * bin_width + self.timestamps[0]
        bin_indices = np.digitize(self.timestamps, bins=bins) - 1
        return bins, bin_indices, n_bins

    def _read_chan_names(self, data):

        def decode_str(int_list):
            ''' interpret an array of integers to a string,
            assuming Unicode.
            '''
            return ''.join([chr(int(i)) for i in int_list])

        return [decode_str(data[ref][:])
                for ref in data['chan_names'][:].squeeze()]

    def _read_spike_times(self, data):
        # ignore index 0, which means unsorted
        n_sorted_units = data['spikes'].shape[0] - 1

        spike_times_dict = {}
        for region in ('M1', 'S1'):
            spike_times_dict[region] = {}

        # iterate over channels
        for idx, channel_name in enumerate(self.chan_names):
            # extract region and channel index from channel name
            region, channel_idx = channel_name.split(' ')
            if region not in ('M1', 'S1'):
                raise ValueError(f'Region {region} is neither M1 nor S1.')

            # iterate over the sorted units only (ignore index 0, which is
            # unsorted).
            for unit in np.arange(n_sorted_units) + 1:
                # extract spike times
                spikes_ref = data['spikes'][unit, idx]
                spike_times = data[spikes_ref][:]

                # if there are no spikes, the data is stored as [0, 0], for
                # some reason
                if np.array_equal(spike_times, np.array([0, 0])):
                    spike_times = np.array([])

                key = f'{int(channel_idx)}_{int(unit)}'
                spike_times_dict[region][key] = spike_times

        return spike_times_dict, n_sorted_units

    def _parse_trials(self):
        # get x, y target position indices with respect to the 8x8 grid
        grids = []
        target_inds = []
        for axis in (0, 1):
            grid = np.unique(self.target_pos[axis, :])
            inds = np.nonzero(
                np.equal(grid.reshape(1, -1),
                         self.target_pos[axis, :].reshape(-1, 1)))[1]
            grids.append(grid)
            target_inds.append(inds)

        # detect target changes
        prev_target_pos = np.concatenate((np.empty((2, 1)) * np.nan,
                                          self.target_pos[:, :-1]), axis=1)
        target_change = np.all(1 - np.equal(self.target_pos, prev_target_pos),
                               axis=0)
        i_start_times = np.nonzero(target_change)[0]

        # construct trials table
        trials_list = []
        for cnt, i_start in enumerate(i_start_times):
            try:
                i_stop = i_start_times[cnt + 1]
            except IndexError:
                i_stop = -1

            row = {'start_time': self.timestamps[i_start],
                   'stop_time': self.timestamps[i_stop],
                   'target_pos_x': self.target_pos[0, i_start],
                   'target_pos_y': self.target_pos[1, i_start],
                   'target_ind_x': target_inds[0][i_start],
                   'target_ind_y': target_inds[1][i_start]}
            trials_list.append(row)
        return pd.DataFrame(trials_list), grids
