import h5py
import numpy as np


class BG():
    def __init__(self, data_path):
        """Processes and provides design/response matrices for the basal
        ganglia recordings during stop/go tasks from the Berkes Lab."""
        self.data_path = data_path
        data = h5py.File(data_path, 'r')

        if 'GoodUnits' in data:
            self.good_units = data['GoodUnits'][:].ravel()
            self.n_good_units = self.good_units.size
        else:
            self.good_units = None

        # populate list of trials
        self.trials, self.bad_trials = self.populate_trials(data)
        self.n_trials = len(self.trials)
        # obtain good trials
        self.good_trials = np.setdiff1d(np.arange(self.n_trials),
                                        self.bad_trials)
        self.n_good_trials = self.good_trials.size
        data.close()

    def populate_trials(self, data):
        """Populate a list of Trial objects given a data object."""
        trials = []
        bad_trials = np.array([])

        # extract trial data
        trial_data = data['Trials']
        n_trials = trial_data['time'].shape[0]

        for idx in range(n_trials):
            print(idx)
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
                spike_rate_data = unit_data['rate']
                n_units = spike_time_data.size
                spike_times = []
                spike_rates = []

                # iterate over units
                for unit in range(n_units):
                    spike_times.append(
                        data[spike_time_data[unit, 0]][:].ravel()
                    )
                    spike_rates.append(
                        data[spike_rate_data[unit, 0]][:].ravel()
                    )

                trial.spike_times = spike_times
                trial.spike_rates = spike_rates

            trials.append(trial)
        return trials, bad_trials

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
        # important times in the trial
        self.t_trial_start = kwargs.get('t_trial_start', None)
        self.t_center_cue = kwargs.get('t_center_cue', None)
        self.t_center_in = kwargs.get('t_center_in', None)
        self.t_side_cue = kwargs.get('t_side_cue', None)
        self.t_center_out = kwargs.get('t_center_out', None)
        self.t_side_in = kwargs.get('t_side_in', None)
        self.t_trial_end = kwargs.get('t_trial_end', None)

        self.trials = kwargs.get('trials', None)
        self.events = kwargs.get('events', None)
        self.spike_times = kwargs.get('spike_times', None)
        self.spike_rates = kwargs.get('spike_rates', None)

    def is_valid_trial(self):
        if self.events is None:
            raise AttributeError('Trial has no events attribute.')
        return self.events['pre_tone'] != 0

    def is_pretone_success(self):
        if self.events is None:
            raise AttributeError('Trial has no events attribute.')

        return self.events['pre_tone'] == 2
