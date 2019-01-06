import h5py
import numpy as np


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
        self.timestamps = data['t'][0, :]
        self.cursor_pos = data['cursor_pos'][:]
        self.target_pos = data['target_pos'][:]
        self.finger_pos = data['finger_pos'][:]
        self.n_sorted_units = data['spikes'].shape[0] - 1

        self.M1_spike_times = {}
        self.S1_spike_times = {}

        # iterate over channels
        for idx, channel in enumerate(data['chan_names'][0]):
            # extract the channel name as a string
            channel_name = data[channel].value.tobytes()[::2].decode()
            # extract region and channel index from channel name
            region, channel_idx = channel_name.split(' ')
            channel_idx = int(channel_idx)

            # iterate over the sorted units only (ignore index 0, which is
            # unsorted).
            for unit in np.arange(self.n_sorted_units) + 1:
                key = str(channel_idx) + '_' + str(unit)
                # extract spike times
                spikes_ref = data['spikes'][unit, idx]
                spike_times = data[spikes_ref][:]

                # if there are no spikes, the data is stored as [0, 0], for
                # some reason
                if np.array_equal(spike_times, np.array([0, 0])):
                    spike_times = np.array([])

                if region == 'M1':
                    self.M1_spike_times[key] = spike_times
                elif region == 'S1':
                    self.S1_spike_times[key] = spike_times
                else:
                    raise ValueError(
                        'Region %s is neither M1 nor S1.' % region)

        data.close()

    def get_response_matrix(
        self, bin_width, region='M1', transform='square_root'
    ):
        """Create the response matrix by binning the spike times.

        Parameters
        ----------
        bin_width : float
            The width of the bins, in seconds.

        region : string
            The region to create a response matrix for ('M1' or 'S1').

        transform : string
            The transform to apply to the spike counts. Available option is
            'square_root'. If None, no transform is applied.

        Returns
        -------
        Y : nd-array, shape (n_trials, n_neurons)
            Response matrix containing (transformed) spike counts of each
            neuron.
        """
        if region == 'M1':
            spike_times = self.M1_spike_times
        elif region == 'S1':
            spike_times = self.S1_spike_times
        else:
            raise ValueError('Region %s is neither M1 nor S1.' % region)

        # calculate number of bins
        n_bins = int(np.ceil(
            (self.timestamps[-1] - self.timestamps[0])/bin_width
        ))
        bins = np.arange(n_bins + 1) * bin_width + self.timestamps[0]

        Y = np.zeros((n_bins, len(spike_times)))

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

        return Y
