import h5py
import numpy as np
from scipy.signal import convolve


class BG():
    def __init__(self, data_path):
        """Processes and provides design/response matrices for the basal
        ganglia recordings during stop/go tasks from the Berkes Lab."""

        self.data_path = data_path

        # populate list of trials
        self.trials, self.bad_trials, self.good_units = \
            self.populate_trials(self.data_path)

        self.n_good_units = self.good_units.size
        self.n_trials = len(self.trials)

        # obtain good trials
        self.good_trials = np.setdiff1d(np.arange(self.n_trials),
                                        self.bad_trials)
        self.n_good_trials = self.good_trials.size

    def get_pre_tone_condition_trials(self, condition=True):
        """Get pre-tone conditions for all trials.

        Parameters
        ----------
        condition : bool
            If True, the pre-tone successes are returned. If False, the
            pre-tone failures are returned.

        trials : ndarray
            The trials satisfying the specified condition.
        """
        trials = np.array([], dtype='int')

        for idx, trial in enumerate(self.trials):
            if condition:
                if trial.is_pretone_success():
                    trials = np.append(trials, idx)
            else:
                if trial.is_pretone_failure():
                    trials = np.append(trials, idx)

        return trials

    def get_binned_spikes(self, trial, unit, sampling_rate=500, bounds=None):
        """Bin spike sequences for a given trial and unit.

        Parameters
        ----------
        trial : int
            The trial index.

        unit : int
            The unit index.

        sampling_rate : int
            The sampling rate of the binning.

        bounds : tuple
            The endpoints, in seconds, to perform the binning.

        Returns
        -------
        binned_spikes : ndarray
            The binned spike counts.

        bins : ndarray
            The timestamps for each point in the firing rate.
        """
        spike_times = self.trials[trial].spike_times[unit]

        if bounds is None:
            bins = self.trials[trial].timestamps
        else:
            bins = np.arange(bounds[0], bounds[1], 1 / sampling_rate)

        binned_spikes, _ = np.histogram(spike_times, bins=bins)

        return binned_spikes, bins

    def get_successful_left_trials(self):
        """Get the trial indices for which the rat successfully moved to the
        left port."""
        indices = np.array([
            idx for idx, trial in enumerate(self.trials)
            if trial.events['go'] == 1
        ])
        return indices

    def get_successful_right_trials(self):
        """Get the trial indices for which the rat successfully moved to the
        right port."""
        indices = np.array([
            idx for idx, trial in enumerate(self.trials)
            if trial.events['go'] == 2
        ])
        return indices

    def get_successful_go_trials(self):
        """Get the trial indices for which the rat successfully completed a
        GO."""
        indices = np.array([
            idx for idx, trial in enumerate(self.trials)
            if trial.is_successful_go()
        ])
        return indices

    def get_successful_stop_trials(self):
        """Get the trial indices for which the rat successfully completed a
        STOP."""
        indices = np.array([
            idx for idx, trial in enumerate(self.trials)
            if trial.is_successful_stop()
        ])
        return indices

    def get_firing_rate(
        self, trial, unit, sampling_rate=500, sigma=0.03, bounds=None,
        kernel_extent=(-2, 2)
    ):
        """Obtain a firing rate estimate using a Gaussian kernel.

        Parameters
        ----------
        trial : int
            The trial index.

        unit : int
            The unit index.

        sampling_rate : int
            The sampling rate of the binning and kernel.

        sigma : float
            The width of the Gaussian kernel.

        bounds : tuple
            The endpoints, in seconds, to extract the firing rate.

        kernel_extent : tuple
            The extent of the Gaussian kernel around the mean.

        Returns
        -------
        firing_rate : ndarray
            The firing rate within the bounds.

        bins : ndarray
            The timestamps for each point in the firing rate.
        """
        spike_times = self.trials[trial].spike_times[unit]

        if bounds is None:
            bins = self.trials[trial].timestamps
        else:
            bins = np.arange(bounds[0], bounds[1], 1 / sampling_rate)
        binned_spikes, _ = np.histogram(spike_times, bins=bins)

        x = np.arange(kernel_extent[0], kernel_extent[1], 1 / sampling_rate)
        kernel = np.exp(-x**2 / (2 * sigma**2)) / np.sqrt(2 * np.pi * sigma**2)

        firing_rate = convolve(binned_spikes, kernel, mode='same')
        return firing_rate, bins[:-1]

    @staticmethod
    def populate_trials(data_path):
        """Populate a list of Trial objects given a data path.

        Parameters
        ----------
        data_path : string
            The path to the dataset.

        Returns
        -------
        trials :

        """
        data = h5py.File(data_path, 'r')

        trials = []
        bad_trials = np.array([])

        # extract trial data
        trial_data = data['Trials']
        n_trials = trial_data['time'].shape[0]

        for idx in range(n_trials):
            trial = Trial()

            # important times in the trial #

            # time of trial start
            trial.t_trial_start = np.asscalar(
                data[trial_data['time'][idx, 0]][:]
            )

            # time of center cue
            trial.t_center_cue = np.asscalar(
                data[trial_data['CenterCueEvent'][idx, 0]][:]
            )

            # time that rat enters center port
            trial.t_center_in = np.asscalar(
                data[trial_data['CenterInEvent'][idx, 0]][:]
            )

            # time of cue to move left or right
            t_side_cue = data[trial_data['SideCueEvent'][idx, 0]][:]
            # check if there was a side cue event
            if t_side_cue.size == 1:
                trial.t_side_cue = np.asscalar(t_side_cue)
            else:
                trial.t_side_cue = None

            # time that rat leaves center port
            trial.t_center_out = np.asscalar(
                data[trial_data['CenterOutEvent'][idx, 0]][:]
            )

            # time that rat enters the side port
            t_side_in = data[trial_data['SideInEvent'][idx, 0]][:]
            if t_side_in.size == 1:
                trial.t_side_in = np.asscalar(t_side_in)
            else:
                trial.t_side_in = None

            # important experimental setup features #
            # center port
            trial.center = np.asscalar(
                data[trial_data['center'][idx, 0]][:]
            )

            # target port
            trial.target = np.asscalar(
                data[trial_data['target'][idx, 0]][:]
            )

            # event codes #
            evt = data[trial_data['Evt'][idx, 0]]['Cond']
            events = {}
            events['pre_tone'] = np.asscalar(data[evt[0, 0]][:])
            events['proactive_inhibition'] = np.asscalar(data[evt[1, 0]][:])
            events['go_cue'] = np.asscalar(data[evt[2, 0]][:])
            events['go_LHMH'] = np.asscalar(data[evt[3, 0]][:])
            events['go'] = np.asscalar(data[evt[4, 0]][:])
            events['go_vs_stop'] = np.asscalar(data[evt[5, 0]][:])
            events['stop'] = np.asscalar(data[evt[6, 0]][:])
            trial.events = events

            # timestamps
            unit_data = data[trial_data['Units'][idx, 0]]

            # sift out bad trials
            if isinstance(unit_data, h5py.Dataset):
                bad_trials = np.append(bad_trials, idx)
                trial.timestamps = None
                trial.spike_times = None
            else:
                # extract time stamps
                trial.timestamps = data[unit_data['times'][0, 0]][:].ravel()

                # extract spike times
                spike_time_data = unit_data['spkTimes']
                n_units = spike_time_data.size
                spike_times = []

                # iterate over units
                for unit in range(n_units):
                    spike_times.append(
                        data[spike_time_data[unit, 0]][:].ravel()
                    )

                trial.spike_times = spike_times

            trials.append(trial)

        # get good units
        if 'GoodUnits' in data:
            good_units = data['GoodUnits'][:].ravel().astype('int') - 1
        else:
            # assume all units are good
            good_units = np.arange(n_units)

        data.close()
        return trials, bad_trials, good_units

    @staticmethod
    def decode_event_condition(event, code):
        """Provides the experimental setting given an event and code.

        Parameters
        ----------
        event : string
            The event category.

        code : int
            The event code.

        Returns
        -------
        definition : string
            A string clarifying the event condition.
        """
        if event == 'pre_tone':
            if code == 0:
                return 'exclude'
            elif code == 1:
                return 'pre_tone_violation'
            elif code == 2:
                return 'pre_tone_success'
            else:
                raise ValueError('Incorrect pre-tone code.')
        elif event == 'proactive_inhibition':
            if code == 0:
                return 'exclude'
            elif code == 1:
                return '50_right_stop'
            elif code == 2:
                return '0_stop'
            elif code == 3:
                return '50_left_stop'
            else:
                raise ValueError('Incorrect proactive inhibition code.')
        elif event == 'go_cue':
            if code == 0:
                return 'exclude'
            elif code == 1:
                return 'left'
            elif code == 2:
                return 'right'
            else:
                raise ValueError('Incorrect go cue code.')
        elif event == 'go_LHMH':
            if code == 0:
                return 'exclude'
            elif code == 1:
                return 'left'
            elif code == 2:
                return 'left_LHMH'
            elif code == 3:
                return 'right'
            elif code == 4:
                return 'right_LHMH'
            else:
                raise ValueError('Incorrect LHMH go code.')
        elif event == 'go':
            if code == 0:
                return 'exclude'
            elif code == 1:
                return 'left'
            elif code == 2:
                return 'right'
            else:
                raise ValueError('Incorrect go code.')
        elif event == 'go_vs_stop':
            if code == 0:
                return 'exclude'
            elif code == 1:
                return 'stop_success'
            elif code == 2:
                return 'stop_fail'
            elif code == 3:
                return 'go_success'
            else:
                raise ValueError('Incorrect go vs. stop code.')
        elif event == 'stop':
            if code == 0:
                return 'exclude'
            elif code == 1:
                return 'stop_success'
            elif code == 2:
                return 'stop_fail'
            else:
                raise ValueError('Incorrect stop code.')
        else:
            raise ValueError('Incorrect event code.')


class Trial():
    def __init__(self, **kwargs):
        """Acts as a struct to store information about a trial in the
        experiment.

        Attributes
        ----------
        t_trial_start : int
            The time, in seconds, when the trial started.

        t_center_cue : int
            The time, in seconds, when the center cue turned on.

        t_center_in : int
            The time, in seconds, when the rat entered the center port.

        t_side_cue : int
            The time, in seconds, when the tone cued to move to the side.
            If the rat failed pre-tone, this attribute is None.

        t_center_out : int
            The time, in seconds, when the rat left the center port.

        t_side_in : int
            The time, in seconds, when the rat entered the side port.
            If the rat did not enter the side port, this attribute is None.

        t_trial_end : int
            The time, in seconds, when the trial ended.

        events : dict
            Contains the event codes detailing the experimental conditions of
            the trial.

        spike_times : list of ndarrays
            List of arrays containing the times, in seconds, that each unit
            spiked.
        """
        self.t_trial_start = kwargs.get('t_trial_start', None)
        self.t_center_cue = kwargs.get('t_center_cue', None)
        self.t_center_in = kwargs.get('t_center_in', None)
        self.t_side_cue = kwargs.get('t_side_cue', None)
        self.t_center_out = kwargs.get('t_center_out', None)
        self.t_side_in = kwargs.get('t_side_in', None)
        self.t_trial_end = kwargs.get('t_trial_end', None)

        self.events = kwargs.get('events', None)
        self.spike_times = kwargs.get('spike_times', None)

    def is_valid_trial(self):
        """Checks whether this Trial object was a valid trial."""
        if self.events is None:
            raise AttributeError('Trial has no events attribute.')
        return self.events['pre_tone'] != 0

    def is_pretone_success(self):
        """Checks whether this Trial object resulted in a pre-tone success."""
        if self.events is None:
            raise AttributeError('Trial has no events attribute.')

        return self.events['pre_tone'] == 2

    def is_pretone_failure(self):
        """Checks whether this Trial object resulted in a pre-tone failure."""
        if self.events is None:
            raise AttributeError('Trial has no events attribute.')

        return self.events['pre_tone'] == 1

    def is_successful_left(self):
        """Checks whether this trial resulted in a successful left to side
        port."""
        if self.events is None:
            raise AttributeError('Trial has no events attribute.')

        return self.events['go'] == 1

    def is_successful_right(self):
        """Checks whether this trial resulted in a successful right to side
        port."""
        if self.events is None:
            raise AttributeError('Trial has no events attribute.')

        return self.events['go'] == 2

    def is_successful_stop(self):
        """Checks whether the rat successfully stopped on this trial."""
        if self.events is None:
            raise AttributeError('Trial has no events attribute.')

        return self.events['go_vs_stop'] == 1

    def is_successful_go(self):
        """Checks whether the rat successfully completed a go trial."""
        if self.events is None:
            raise AttributeError('Trial has no events attribute.')

        return self.events['go_vs_stop'] == 3
