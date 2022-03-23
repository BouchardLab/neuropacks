import h5py
import numpy as np
import pandas as pd

from neuropacks.utils.binning import bin_spike_times, create_bins


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

        cursor_pos : nd-array, shape (2, n_timestamps)
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

        trials : pandas DataFrame
            Each row is a trial, i.e., interval with a fixed target position.
            Columns ('start_time', 'stop_time', 'target_pos_x', 'target_pos_y',
                     'target_ind_x', 'target_ind_y').

        target_grid : list
            List of two ndarrays, [x_grid, y_grid].
            Each of x_grid and y_grid has 8 values by experimental design.
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
        pos_binned : ndarray
            The mean x, y position in each bin. Has shape (n_timestamps, 2).
        """
        return self.bin_timeseries(self.cursor_pos, bin_width=bin_width)

    def bin_timeseries(self, data_ts, bin_width=0.5):
        """Bin the provided timeseries data along the time axis,
        according to a bin width.

        Parameters
        ----------
        bin_width : float
            The width of the bin, in seconds.

        Returns
        -------
        data_binned : ndarray, shape (n_bins, 2)
            The mean x, y position in each bin.
        """
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
        Y : nd-array, shape (n_bins, n_units)
            Response matrix containing (transformed) spike counts of each
            neuron (single unit).
        """
        if transform is None:
            # no transform; just pass None to bin_spike_times
            pass
        elif transform in ('sqrt', 'square_root'):
            transform = np.sqrt
        else:
            raise ValueError(f'Invalid transform {transform} for this dataset.')

        try:
            spike_times = self.spike_times[region]
        except KeyError:
            raise ValueError(f'Region {region} is neither M1 nor S1.')

        bins, _, n_bins = self._bin_times(bin_width)
        spike_times_array = [spikes for key, spikes in spike_times.items()]
        Y = bin_spike_times(spike_times_array, bins, apply_transform=transform)
        return Y

    def get_binned_times(self, bin_width=0.5):
        """Bin time according to a bin width, and return the center values.

        Parameters
        ----------
        bin_width : float
            The width of the bin, in seconds.

        Returns
        -------
        times_bin_center : ndarray, shape (n_bins, )
            The representative time value (center-of-bin) for each bin.
        """
        bins, _, _ = self._bin_times(bin_width)
        times_bin_center = (bins[1:] + bins[:-1]) / 2
        return times_bin_center

    def _bin_times(self, bin_width):
        bins = create_bins(self.timestamps[0], self.timestamps[-1], bin_width)
        n_bins = len(bins) - 1
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
