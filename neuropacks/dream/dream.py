import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio


class DREAM:
    def __init__(self, data_path):
        """Processes and provides design/response matrices for the primary
        visual cortex 11 dataset from CRCNS.

        Parameters
        ----------
        data_path : string
            The path to the DREAM dataset.

        Attributes
        ----------
        data_path : string
            The path to the DREAM dataset.

        data : MAT object
            The MAT object containing the recorded data in the the DREAM
            dataset.

        subjects : MAT object

        n_subjects : int
            The number of subjects in this dataset.

        n_trials : list of ints
            The number of trials in each recording session.

        n_neurons : list of ints
            The number of neurons in each recording session.

        targets : nd-array, shape (n_subjects, 8, 2)
            The x, y locations of the 8 targets for each subject in the
            experiment.

        centers : nd-array, shape (n_subjects, 2)
            The x, y locations of the centers for each subject in the
            experiment.

        valid_trials : dict of ints to nd-arrays
            A dictionary containing the valid trials (where a target was
            displayed) for each subject. The keys correspond to each subject
            index while the values correspond to nd-arrays containing the
            indices of the valid trials.
        """
        self.data_path = data_path
        # grab data object
        self.data = scio.loadmat(data_path, struct_as_record=False)
        # grab subjects
        self.subjects = self.data['Subject'].ravel()
        self.n_subjects = self.subjects.size
        # extract number of trials
        self.n_trials = [subject.Trial.size for subject in self.subjects]
        # extract number of neurons
        self.n_neurons = [
            subject.Trial[0, 0].Neuron.size for subject in self.subjects
        ]
        # grab targets
        self.targets = np.zeros((self.n_subjects, 8, 2))
        self.centers = np.zeros((self.n_subjects, 2))
        for subject_idx in range(self.n_subjects):
            target, center = self.get_targets(subject_idx=subject_idx)
            self.targets[subject_idx] = target
            self.centers[subject_idx] = center
        # extract valid trials indices
        self.valid_trials = {
            subject_idx: np.argwhere(
                self.get_valid_trial_idx(subject_idx)
            ).ravel()
            for subject_idx in range(self.n_subjects)
        }
        self.n_valid_trials = [
            len(self.valid_trials[idx]) for idx in self.valid_trials.keys()
        ]

    def get_subject(self, subject_idx):
        """Returns the subject MATLAB struct given an index.

        Parameters
        ----------
        subject_idx : int
            The index of the desired subject.

        Returns
        -------
        subject : MATLAB struct
            The subject MATLAB struct.
        """
        # check if a valid subject index was provided
        if (subject_idx >= self.n_subjects) or (subject_idx < 0):
            raise ValueError("Subject index is not valid.")
        else:
            subject = self.subjects[subject_idx]
        return subject

    def get_trials(self, subject_idx):
        """Returns an array of trial Matlab structs for a given subject.

        Parameters
        ----------
        subject_idx : int
            The index of the desired subject.

        Returns
        -------
        trials : array of MATLAB structs
            An array of Trial structs corresponding to the desired subject.
        """
        # extract subject
        subject = self.get_subject(subject_idx)
        # extract trials
        trials = subject.Trial.ravel()
        return trials

    def get_trial(self, subject_idx, trial_idx):
        """Returns a desired trial performed by a specified subject.

        Parameters
        ----------
        subject_idx : int
            the index of the desired subject.

        trial_idx : int
            the index of the desired trial.
        """
        # obtain all trials
        trials = self.get_trials(subject_idx)
        n_trials = trials.size
        # check if desired trial index is valid
        if (trial_idx >= n_trials) or (trial_idx < 0):
            raise ValueError('Trial index is not valid.')
        else:
            trial = trials[trial_idx]
        return trial

    def get_targets(self, subject_idx):
        """Get the Cartesian coordinates for the targets in a trial.

        Parameters
        ----------
        subject_idx : int
            the index of the desired subject.

        Returns
        -------
        targets : nd-array, shape (8, 2)
            The x, y positions of the eight non-center targets.

        center : nd-array, shape (1, 2)
            The x, y position of the center target.
        """
        # get subject
        subject = self.get_subject(subject_idx)
        # extract trials
        trials = subject.Trial.ravel()
        n_trials = trials.size
        # initialize targets
        targets = np.array([[0, 0, 0]])

        # iterate over trials
        for trial_idx in range(n_trials):
            # grab trial
            trial = trials[trial_idx]
            # extract indices for which target is on
            target_indices = ~np.isnan(trial.TargetPos)
            # grab unique target values
            target = np.unique(trial.TargetPos[target_indices[:, 0]], axis=0)
            # add the new targets on
            targets = np.concatenate((targets, target), axis=0)

        # extract only unique targets
        targets = np.unique(targets[1:, :2], axis=0)
        # get center
        center = np.mean(targets, axis=0)
        # remove center target
        targets = np.array([
            target for target in targets if not np.allclose(target, center)
        ])
        return targets, center

    def get_timestamps_for_trial(self, subject_idx, trial_idx):
        """Return an array of timestamps for a given trial.

        Parameters
        ----------
        subject_idx : int
            The index of the desired subject.

        trial_idx : int
            The index of the desired trial.

        Returns
        -------
        timestamps : nd-array
            An array containing the time, in seconds, of each cursor recording.
        """
        # grab specific trial
        trial = self.get_trial(subject_idx, trial_idx)
        # grab timestamps
        timestamps = trial.Time.ravel()
        return timestamps

    def get_stim_onset_for_trial(self, subject_idx, trial_idx):
        """Get the index and timestamp for when the stimulus target turns on.

        Parameters
        ----------
        subject_idx : int
            The index of the desired subject.

        trial_idx : int
            The index of the desired trial.

        Returns
        -------
        idx: int or None
            The index of the stimulus onset; None if trial does not contain a
            non-center target

        timestamp : float
            The timestamp, in seconds, of stimulus onset.

        target : nd-array, shape (1, 2)
            The x,y coordinates for the reach target on the specified trial.
        """
        # extract trial
        trial = self.get_trial(subject_idx, trial_idx)
        # grab center target
        center = self.centers[subject_idx]
        # get timestamps
        timestamps = self.get_timestamps_for_trial(
            subject_idx=subject_idx,
            trial_idx=trial_idx
        )
        # get position of targets at each timestamp (remove extraneous z
        # coordinate)
        target_position = trial.TargetPos[:, :2]

        # iterate over targets
        for time_idx, target in enumerate(target_position):
            # check if there is no target on
            if not np.all(np.isnan(target)):
                # check if target is not equal to center
                if not np.all(np.isclose(center, target)):
                    # if we're here, this is the first time a non-center target
                    # has turned on
                    return time_idx, timestamps[time_idx], target

        # if we reached here, the trial is not valid
        return None

    def get_valid_trial_idx(self, subject_idx):
        """Returns the valid trials for a given subject.

        Parameters
        ----------
        subject_idx : int
            The index of the desired subject.

        Returns
        -------
        valid_trials : nd-array of bools
            A numpy array with flags indicating which trials are valid.
        """
        n_trials = self.n_trials[subject_idx]
        valid_trials = np.zeros(n_trials, dtype='bool')

        for trial_idx in range(n_trials):
            # check whether the stim onset exists
            stim_onset = self.get_stim_onset_for_trial(
                subject_idx=subject_idx,
                trial_idx=trial_idx
            )
            if stim_onset is not None:
                valid_trials[trial_idx] = True

        return valid_trials

    def get_target_for_trial(self, subject_idx, trial_idx):
        """Returns the target of the reach in a specified trial.

        Parameters
        ----------
        subject_idx : int
            The index of the desired subject.

        trial_idx : int
            The index of the desired trial.

        Returns
        -------
        target : nd-array, shape (2)
            The x,y coordinates for the reach target on the specified trial.
            None if no reach target on trial.
        """
        stim_onset = self.get_stim_onset_for_trial(
            subject_idx=subject_idx,
            trial_idx=trial_idx
        )
        # check whether a target flashed on this trial
        if stim_onset is not None:
            target = stim_onset[2]
            return target
        else:
            return None

    def get_angle_for_trial(self, subject_idx, trial_idx):
        """Returns the angle of the reach in a specified trial.

        Parameters
        ----------
        subject_idx : int
            The index of the desired subject.

        trial_idx : int
            The index of the desired trial.

        Returns
        -------
        angle : float
            The reach of the angle, constrained to [0, 360) degrees.
        """
        reach_target = self.get_target_for_trial(subject_idx, trial_idx)
        center = self.centers[subject_idx]
        # only calculate angle if there's a valid reach target
        if reach_target is not None:
            angle = self.calculate_angle(center, reach_target)
            return angle
        else:
            # if we're here, there's no non-center target in this trial
            return reach_target

    def get_hand_pos_for_trial(self, subject_idx, trial_idx):
        """Returns the hand position over a trial.

        Parameters
        ----------
        subject_idx : int
            The index of the desired subject.

        trial_idx : int
            The index of the desired trial.

        Returns
        -------
        hand_position : numpy array
            The x,y coordinates of the hand position
        """
        # extract trial
        trial = self.get_trial(subject_idx, trial_idx)
        # extract hand position
        hand_position = trial.HandPos[:, :2]
        return hand_position

    def get_movement_onset_for_trial(
        self, subject_idx, trial_idx, tol=23.5, consecutive=5
    ):
        """Returns the index, timestamp, and hand position corresponding to
        movement onset for a given trial.

        Parameters
        ----------
        subject_idx : int
            The index of the desired subject.

        trial_idx : int
            The index of the desired trial.

        tol : float, default 23.5
            The tolerance of movement onset, in degrees. The default value
            (23.5)corresponds to the index at which the angle is closest to the
            angle of reach.

        consecutive : int, default 5
            The number of consecutive timepoints that must satisfy the
            criterion in order to qualify as a movement onset.

        Returns
        -------
        index : int or None
            The index of the movement onset. None if the trial is invalid or
            there is no valid starting index within the specified parameters.

        timestamp : float
            The timestamp corresponding to movement onset. None if the trial is
            invalid or there is no valid starting index within the specified
            parameters.

        hand_position : tuple of two floats
            Contains the x,y coordinates at movement onset. None if the trial
            is invalid or there is no valid starting index within the specified
            parameters.
        """
        # get stim onset
        stim_onset = self.get_stim_onset_for_trial(
            subject_idx=subject_idx,
            trial_idx=trial_idx
        )

        if stim_onset is not None:
            stim_onset_idx, stim_onset_time, target = stim_onset

            # get hand position for trial
            hand_position = self.get_hand_pos_for_trial(
                subject_idx=subject_idx,
                trial_idx=trial_idx
            )

            # get timestamps for trial
            timestamps = self.get_timestamps_for_trial(
                subject_idx=subject_idx,
                trial_idx=trial_idx
            )

            # get angle for trial
            angle = self.get_angle_for_trial(
                subject_idx=subject_idx,
                trial_idx=trial_idx
            )

            # set counter for consecutive number of points
            counter = 0

            # iterate over time points after stim onset
            for time_idx in range(stim_onset_idx, timestamps.size):
                start = hand_position[time_idx - 1]
                end = hand_position[time_idx]
                cur_angle = self.calculate_angle(start, end)

                if np.abs(cur_angle - angle) < tol:
                    counter += 1
                    if counter >= consecutive:
                        true_onset_idx = time_idx - consecutive
                        return (true_onset_idx,
                                timestamps[true_onset_idx],
                                hand_position[true_onset_idx])
                else:
                    counter = 0
            return (stim_onset_idx,
                    stim_onset_time,
                    hand_position[stim_onset_idx])
        else:
            return None, None, None

    def get_speed_for_trial(self, subject_idx, trial_idx, dT, **kwargs):
        """Returns the neuron MATLAB structs for a given subject and trial.

        Parameters
        ----------
        subject_idx : int
            The index of the desired subject.

        trial_idx : int
            The index of the desired trial.

        dT : tuple of two floats
            The offsets from the movement onset over which to calculate the
            speed.

        Returns
        -------
        speed : float
            The average speed over the specified bound.
        """
        # get hand position for trial
        hand_position = self.get_hand_pos_for_trial(
            subject_idx=subject_idx,
            trial_idx=trial_idx
        )

        # get timestamps for trial
        timestamps = self.get_timestamps_for_trial(
            subject_idx=subject_idx,
            trial_idx=trial_idx
        )

        # get movement onset
        onset_idx, onset_time, hand_pos_onset = \
            self.get_movement_onset_for_trial(
                subject_idx=subject_idx,
                trial_idx=trial_idx,
                tol=kwargs.get('tol', 23.5),
                consecutive=kwargs.get('consecutive', 5)
            )

        # set time bounds
        t_bounds = (onset_time + dT[0], onset_time + dT[1])
        # find nearest indices to time bounds
        t_bounds_idx = (
            np.argmin(np.abs(timestamps - t_bounds[0])),
            np.argmin(np.abs(timestamps - t_bounds[1]))
        )
        # calculate distance
        distance = np.sqrt(np.sum(
            (hand_position[t_bounds_idx[1]] -
             hand_position[t_bounds_idx[0]])**2
        ))
        # calculate speed
        speed = distance/(dT[1] - dT[0])
        return speed

    def get_neurons_for_trial(self, subject_idx, trial_idx):
        """Returns the neuron MATLAB structs for a given subject and trial.

        Parameters
        ----------
        subject_idx : int
            The index of the desired subject.

        trial_idx : int
            The index of the desired trial.

        Returns
        -------
        neurons : array of Matlab structs
            An array of MATLAB structs containing information about the
            neurons.
        """
        # extract trial
        trial = self.get_trial(subject_idx, trial_idx)
        # get neurons
        neurons = trial.Neuron.ravel()
        return neurons

    def get_spikes_for_trial(self, subject_idx, trial_idx):
        """Returns a dictionary of spike times over the neural population for
        a given subject and trial.

        Parameters
        ----------
        subject_idx : int
            The index of the desired subject.

        trial_idx : int
            The index of the desired trial.

        Returns
        -------
        spikes : dict
            Dictionary with keys equal to neuron indices and values equal
            to numpy arrays containing spike times in seconds.
        """
        # get neurons
        neurons = self.get_neurons_for_trial(
            subject_idx=subject_idx,
            trial_idx=trial_idx
        )
        # create spike dictionary
        spikes = {
            neuron_id: spike_times.Spike.ravel()
            for neuron_id, spike_times in enumerate(neurons)
        }
        return spikes

    def get_spikes_for_trial_and_neuron(
        self, subject_idx, trial_idx, neuron_idx
    ):
        """Returns the spike times for a given subject, trial, and neuron.

        Parameters
        ----------
        subject_idx : int
            The index of the desired subject.

        trial_idx : int
            The index of the desired trial.

        neuron_idx : int
            The index of the desired neuron.

        Returns
        -------
        spikers_for_neuron : nd-array
            An array containing the spike times for a given neuron.
        """
        # grab all spikes
        spikes = self.get_spikes_for_trial(
            subject_idx=subject_idx,
            trial_idx=trial_idx
        )
        if (neuron_idx > self.n_neurons[subject_idx]) or (neuron_idx < 0):
            raise ValueError('Invalid neuron index.')
        # grab spikes for specified neuron
        spikes_for_neuron = spikes[neuron_idx]
        return spikes_for_neuron

    def get_spike_counts_for_trial(
        self, subject_idx, trial_idx, onset='movement', dT=(0.1, 0.3),
        tol=23.5, consecutive=5
    ):
        """Returns the spike count for a given neuron on a chosen subject and
        trial.

        Parameters
        ----------
        subject_idx : int
            The index of the desired subject.

        trial_idx : int
            The index of the desired trial.

        onset : string
            The frame of reference for choosing the bin to calculate spike
            counts. Valid options are 'movement' or 'stimulus'.

        dT : tuple of two floats
            Contains the time bounds, in seconds, relative to the onset for
            which the spike count will be calculated in.

        tol : float, default 23.5
            The tolerance of movement onset, in degrees. The default value
            (23.5)corresponds to the index at which the angle is closest to the
            angle of reach.

        consecutive : int, default 5
            The number of consecutive timepoints that must satisfy the
            criterion in order to qualify as a movement onset.

        Returns
        -------
        spike_counts : numpy array
            an array containing the spike counts for each neuron
        """
        # get onset time
        if onset == 'stimulus':
            onset_idx, onset_time, _ = self.get_stim_onset_for_trial(
                subject_idx=subject_idx,
                trial_idx=trial_idx)

        elif onset == 'movement':
            onset_idx, onset_time, _ = self.get_movement_onset_for_trial(
                subject_idx=subject_idx,
                trial_idx=trial_idx,
                tol=tol,
                consecutive=consecutive
            )

        else:
            raise ValueError("Incorrect Onset Value.")

        # establish bounds for spike bin
        bounds = (onset_time + dT[0], onset_time + dT[1])

        # get spikes
        spikes = self.get_spikes_for_trial(subject_idx, trial_idx)
        # get spike counts in bin
        spike_counts = np.zeros(len(spikes))

        for neuron_id, spike_times in spikes.items():
            # determine which spikes are within the bounds
            valid_spikes = (
                (spike_times >= bounds[0]) & (spike_times <= bounds[1])
            )
            # obtain the spike counts
            spike_counts[neuron_id] = np.count_nonzero(valid_spikes)

        return spike_counts

    def get_design_matrix(self, subject_idx, form='angle', **kwargs):
        """Create the design matrix.

        Parameters
        ----------
        subject_idx : int
            The index of the desired subject.

        form : string
            The type of design matrix to use. Valid options are "angle",
            "cosine", "cosine_speed", "one_hot", and "speed_hot".

        Returns
        -------
        X : nd-array, shape (n_trials, n_features)

        """
        # get valid trials
        valid_trials = self.valid_trials[subject_idx]

        # set up angle matrix
        angles = np.zeros(valid_trials.size)
        for idx, trial in enumerate(valid_trials):
            angles[idx] = self.get_angle_for_trial(
                subject_idx=subject_idx,
                trial_idx=trial
            )

        # establish design matrix #
        # just return the angle for each trial
        if form == 'angle':
            X = angles

        # the first column is the cosine of the angle, the second is the sine
        # of the angle
        elif form == 'cosine':
            X = np.zeros((angles.size, 2))
            X[:, 0] = np.cos(np.deg2rad(angles))
            X[:, 1] = np.sin(np.deg2rad(angles))

        # same as cosine, but there's a column for speed and basis
        elif form == 'cosine_speed':
            # functions are tuned for speed
            X = np.zeros((angles.size, 3))
            for idx, angle in enumerate(angles):
                speed = self.get_speed_for_trial(
                    subject_idx=subject_idx,
                    trial_idx=valid_trials[idx],
                    dT=kwargs.get('dT', (0.1, 0.3)),
                    **kwargs
                )
                X[idx, 0] = speed
                X[idx, 1] = speed * np.cos(np.deg2rad(angle))
                X[idx, 2] = speed * np.sin(np.deg2rad(angle))

        # one-hot encoding of the angle, enabling 8 features to be learned
        elif form == 'one_hot':
            X = np.zeros((angles.size, 8))

            for idx, angle in enumerate(angles):
                X[idx, int(angle/45.)] = 1.

        # one-hot encoding of the angle, but the magnitude is given by the
        # magnitude of the speed
        elif form == 'speed_hot':
            X = np.zeros((angles.size, 8))

            for idx, angle in enumerate(angles):
                speed = self.get_speed_for_trial(
                    ubject_idx=subject_idx,
                    trial_idx=valid_trials[idx],
                    dT=kwargs.get('dT', (0.1, 0.3)),
                    **kwargs
                )
                X[idx, int(angle/45.)] = speed

        else:
            raise ValueError('Incorrect form of design matrix specified.')

        return X

    def get_response_matrix(
        self, subject_idx, transform=None, active_bound=None, **kwargs
    ):
        """Create the response matrix.

        Parameters
        ----------
        subject_idx : int
            The index of the desired subject.

        transform : string
            The transform to apply to the spike counts. Available option is
            "square_root". If None, no transform is applied.

        active_bound : float
            The threshold, in firing rate, to include a neuron in the response
            matrix (i.e. extract "active" neurons). If None, no thresholding
            is performed.

        Returns
        -------
        Y : nd-array, shape (n_trials, n_neurons)
            Response matrix containing (transformed) spike counts of each
            neuron.
        """
        # get valid trials
        valid_trials = self.valid_trials[subject_idx]
        # get number of neurons
        n_neurons = self.n_neurons[subject_idx]
        # set up response matrix
        Y = np.zeros((valid_trials.size, n_neurons))
        # grab angles
        for idx, trial in enumerate(valid_trials):
            Y[idx] = self.get_spike_counts_for_trial(
                subject_idx=subject_idx,
                trial_idx=trial,
                onset=kwargs.get('onset', 'movement'),
                dT=kwargs.get('dT', (0.1, 0.3)),
                tol=kwargs.get('tol', 23.5),
                consecutive=kwargs.get('consecutive', 5)
            )
        # extract "active" neurons
        if active_bound is not None:
            bounds = kwargs.get('dT', (0.1, 0.3))
            dT = bounds[1] - bounds[0]
            active_indices = np.mean(Y, axis=0) / dT > active_bound
            Y = Y[:, active_indices]

        # perform any desired transforms
        if transform is not None:
            if transform == 'square_root':
                Y = np.sqrt(Y)
            else:
                raise ValueError("Transform %s is not valid." % transform)
        return Y

    def plot_trial(self, subject_idx, trial_idx, ax=None):
        """Plot the targets and movement for a given trial.

        Parameters
        ----------
        subject_idx : int
            The index of the desired subject.

        trial_idx : string
            The index of the desired trial.

        ax : Matplotlib axis
            Axis to plot the trial. If None, a new axis is created.

        Returns
        -------
        ax : Matplotlib axis
            An axis containing the target and hand position plot.
        """
        # if no axis provided, create a new one
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        # grab center
        center = self.centers[subject_idx]

        # grab target
        reach_target = self.get_target_for_trial(
            subject_idx=subject_idx,
            trial_idx=trial_idx
        )

        # grab hand position
        hand_position = self.get_hand_pos_for_trial(
            subject_idx=subject_idx,
            trial_idx=trial_idx
        )
        hand_pos_x = hand_position[:, 0]
        hand_pos_y = hand_position[:, 1]

        # plot center
        ax.scatter(center[0], center[1], color='k', s=100, zorder=10)

        # plot targets
        for target in self.targets[subject_idx]:
            if reach_target is not None:
                if np.allclose(target, reach_target):
                    c = 'k'
                else:
                    c = 'lightgray'
            else:
                c = 'lightgray'
            ax.scatter(target[0], target[1], color=c, s=100, zorder=10)

        # plot hand position
        ax.scatter(
            hand_pos_x,
            hand_pos_y,
            c=np.cos(np.arange(hand_pos_x.size) * (np.pi/hand_pos_x.size)),
            s=30
        )

        # set bounds
        ax.set_xlim([center[0] - 0.15, center[0] + 0.15])
        ax.set_ylim([center[1] - 0.15, center[1] + 0.15])

        # set colors
        ax.set_facecolor('white')
        for spine in ax.spines:
            ax.spines[spine].set_edgecolor('k')

        # labels
        ax.set_title(r'\textbf{Trial %s}' % trial_idx, fontsize=30)
        ax.set_xlabel(r'\textbf{X Position}', fontsize=30)
        ax.set_ylabel(r'\textbf{Y Position}', fontsize=30)

        # remove axis labels
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_yticklabels([])

        return ax

    @staticmethod
    def calculate_angle(start, end):
        """Calculate the angle of a vector connecting two points with respect
        to the positive x-axis.

        Parameters
        ----------
        start : nd-array, shape (2)
            The x,y coordinates for start position.

        end : nd-array, shape (2)
            The x,y coordinates for end position.

        Returns
        -------
        angle : float
            The angle of the vector connecting start to end.
        """
        # vector from center to target
        vec = end - start
        # clip any tiny values
        vec[np.isclose(vec, 0)] = 0
        # calculate angle
        angle = np.round(np.arctan2(vec[1], vec[0]) * 180/np.pi)
        # constrain angle to range [0, 360)
        if angle < 0:
            angle += 360.
        return angle

    @staticmethod
    def get_tuning_modulation_and_preference(self, form, tuning_coefs):
        """Extracts the tuning modulation and preference from a set
        of tuning coefficients.

        Parameters
        ----------
        form : string
            The type of design matrix used. Valid options are "angle",
            "cosine", "cosine_speed", "one_hot", and "speed_hot".

        tuning_coefs : nd-array, shape (n_neurons, n_features)
            The coefficients describing the tuning of each neuron.

        Returns
        -------
        modulations : nd-array of floats
            The modulation (min-to-max distance) for each neuron.

        preferences : nd-array of floats
            The preference (location of tuning maximum) for each neuron.
        """

        if form == 'cosine':
            c1 = tuning_coefs[:, 0]
            c2 = tuning_coefs[:, 1]
            preferences = np.arctan2(c2, c1) * (180/np.pi)
            preferences[preferences < 0] += 360
            preferences_rad = np.deg2rad(preferences)
            modulations = \
                (c2 - c1)/(np.sin(preferences_rad) - np.cos(preferences_rad))

        elif form == 'cosine_speed':
            c1 = tuning_coefs[:, 1]/tuning_coefs[:, 0]
            c2 = tuning_coefs[:, 2]/tuning_coefs[:, 0]
            preferences = np.arctan2(c2, c1) * (180/np.pi)
            preferences[preferences < 0] += 360
            preferences_rad = np.deg2rad(preferences)
            modulations = \
                (c2 - c1)/(np.sin(preferences_rad) - np.cos(preferences_rad))

        elif form == 'one_hot':
            preferences = 45 * np.argmax(tuning_coefs, axis=1)
            modulations = \
                np.max(tuning_coefs, axis=1) - np.min(tuning_coefs, axis=1)

        else:
            raise ValueError('Form %s is not available.' % form)

        return modulations, preferences
