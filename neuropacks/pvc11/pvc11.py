import numpy as np
from scipy.io import loadmat


class PVC11():
    def __init__(self, data_path):
        """Processes and provides design/response matrices for the primary
        visual cortex 11 dataset from CRCNS.

        Parameters
        ----------
        data_path : string
            The path to the PVC11 dataset.

        Attributes
        ----------
        data_path : string
            The path to the PVC11 dataset.
        events : nd-array, shape (n_neurons, n_stimuli, n_trials)
            A numpy array containing the spiking responses for each neuron
            to each stimulus on each trial.
        n_neurons : int
            The number of neurons in the experiment.
        n_stimuli : int
            The number of stimuli shown in the experiment.
        n_trials : int
            The number of trials for each stimulus in the experiment.
        spike_counts : nd-array, shape (n_neurons, n_stimuli, n_trials)
            A numpy array containing the spiking count for each neuron
            to each stimulus on each trial.
        """
        self.data_path = data_path
        self.load_data()
        self.get_spike_counts()

    def load_data(self):
        """Loads the PVC-11 data."""
        # Load data
        data = loadmat(self.data_path, struct_as_record=False)
        # Extract events
        self.events = data['data'][0, 0].EVENTS
        # Extract experiment parameters
        self.n_neurons, self.n_stimuli, self.n_trials = self.events.shape

    def get_spike_counts(self):
        """Get spike counts from dataset.

        Returns
        -------
        spike_counts : nd-array, shape (n_neurons, n_stim, n_trials)
            An array containing spike counts over all neurons and trials.
        """
        self.spike_counts = np.zeros(
            (self.n_neurons, self.n_stimuli, self.n_trials)
        )
        # Iterate over all trials, stimuli, and neurons
        for neuron in range(self.n_neurons):
            for stim in range(self.n_stimuli):
                for trial in range(self.n_trials):
                    # Compute all spike counts in the trial
                    self.spike_counts[neuron, stim, trial] = \
                        self.events[neuron, stim, trial].size

        return self.spike_counts

    def get_angles(self):
        """Obtain the unique angles of gratings shown during the experiment.

        Returns
        -------
        angles : np.ndarray
            The unique angles, in degrees, of drifting gratings shown in the
            experiment.
        """
        angles = np.linspace(0, 360, self.n_stimuli + 1)[:-1]
        return angles

    def get_design_matrix(self, form='angle', angles=None, **kwargs):
        """Create design matrix according to a specified form.

        Parameters
        ----------
        form : string
            The structure of the design matrix.
        angles : ndarray
            The angles to obtain a design matrix for. If None, all angles will
            be used.

        Returns
        -------
        X : nd-array, shape (n_trials, n_features)
            The design matrix.
        """
        if angles is None:
            unique_angles = self.get_angles()
            angles = np.repeat(unique_angles, self.n_trials)
        else:
            unique_angles = np.unique(angles)

        if form == 'angle':
            # The angles for each trial; no extra dimension required
            X = angles
        elif form == 'label':
            X = (angles/30).astype('int64')
        elif form == 'cosine':
            X = np.zeros((angles.size, 2))
            X[:, 0] = np.cos(np.deg2rad(angles))
            X[:, 1] = np.sin(np.deg2rad(angles))
        elif form == 'cosine2':
            X = np.zeros((angles.size, 2))
            X[:, 0] = np.cos(2 * np.deg2rad(angles))
            X[:, 1] = np.sin(2 * np.deg2rad(angles))
        elif form == 'one_hot':
            X = np.zeros((angles.size, unique_angles.size))

            for idx, angle in enumerate(angles):
                angle_idx = np.asscalar(np.argwhere(unique_angles == angle))
                X[idx, angle_idx] = 1
        elif form == 'gbf':
            n_bf = kwargs.get('n_bf', 20)
            lower_bound = kwargs.get('lower_bound', 15)
            upper_bound = kwargs.get('upper_bound', 345)
            var = kwargs.get('var', 25)

            means = np.linspace(lower_bound, upper_bound, n_bf)
            norm = 1./np.sqrt(2 * np.pi * var)

            X = np.zeros((angles.size, n_bf))

            for idx, angle in enumerate(angles):
                X[idx] = norm * np.exp(
                    -(angle - means)**2 / (2 * var)
                )
        elif form == 'cbf':
            n_bf = kwargs.get('n_bf', 30)
            lower_bound = kwargs.get('lower_bound', 0)
            upper_bound = kwargs.get('upper_bound', 360)

            means = np.linspace(lower_bound, upper_bound, n_bf)

            X = np.zeros((angles.size, n_bf))

            for idx, angle in enumerate(angles):
                X[idx] = np.cos(2 * np.deg2rad(angle - means))
        else:
            raise ValueError("Incorrect design matrix form specified.")

        return X

    def get_response_matrix(self, transform=None):
        """Calculates response matrix.

        The ordering for the trials in the response matrix is given by:
            stimulus -> trials
        where stimuli are ordered by increasing angle.

        Parameters
        ----------
        transform : string
            Post-processing tranform to apply to spike counts. Default is not
            to apply a transform.

        Returns
        -------
        Y : nd-array, shape (n_responses, n_neurons)
            An array containing the spike counts for each neuron over all the
            trials.
        """

        Y = np.reshape(
            self.spike_counts,
            (self.n_neurons, self.n_stimuli * self.n_trials)
        ).T

        # Apply desired transform
        if transform == 'square_root':
            Y = np.sqrt(Y)
        elif transform == 'log':
            Y = np.log(Y)
        else:
            if transform is not None:
                raise ValueError("Transform %s is not recognized." % transform)

        return Y

    def get_tuning_curve(
        self, form, tuning_coefs, intercept=None, angles=None
    ):
        """Gets the tuning curve given a set of tuning coefficients.

        Parameters
        ----------
        form : string
            The type of design matrix used. Valid options are "angle",
            "cosine", "cosine_speed", "one_hot", and "speed_hot".
        tuning_coefs : nd-array, shape (n_features)
            The coefficients describing the tuning of the neuron.
        intercept : float, optional
            The intercept.
        angles : ndarray
            The angles over which to calculate the tuning curve. If None, a
            default set is used.

        Returns
        -------
        tuning_curve : ndarray
            The responses across the set of angles.
        """
        # If no angles are provided, calculate a set
        if angles is None:
            angles = self.get_angles()
            if form != 'one_hot':
                angles = np.linspace(angles[0], angles[-1], 1000)

        # Calculate design matrix for angles
        X = self.get_design_matrix(form=form, angles=angles)

        # Calculate tuning curve
        tuning_curve = np.dot(X, tuning_coefs)
        if intercept is not None:
            tuning_curve += intercept

        return angles, tuning_curve

    def get_tuning_modulation_and_preference(self, tuning_coefs, form='cosine2'):
        """Extracts the tuning modulation and preference from a set
        of tuning coefficients.

        Parameters
        ----------
        tuning_coefs : nd-array, shape (n_neurons, n_features)
            The coefficients describing the tuning of each neuron.
        form : string
            The type of design matrix used. Valid options are "angle",
            "cosine", "cosine_speed", "one_hot", and "speed_hot".

        Returns
        -------
        modulations : nd-array of floats
            The modulation (min-to-max distance) for each neuron.
        preferences : nd-array of floats
            The preference (location of tuning maximum) for each neuron.
        """
        if tuning_coefs.ndim == 1:
            tuning_coefs = tuning_coefs[np.newaxis]

        if form == 'cosine':
            # Splt up cosine coefficients
            c1 = tuning_coefs[..., 0]
            c2 = tuning_coefs[..., 1]
            # Get preferences in degrees and restrict range to [0, 360)
            preferences = np.arctan2(c2, c1) * (180/np.pi)
            preferences[preferences < 0] += 360
            preferences_rad = np.deg2rad(preferences)
            # Calculate modulations
            modulations = 2 * (c2 - c1) / (np.sin(preferences_rad)
                                           - np.cos(preferences_rad))

        elif form == 'cosine2':
            # Split up cosine coefficients
            c1 = tuning_coefs[..., 0]
            c2 = tuning_coefs[..., 1]
            # Get preferences in degrees and restrict range to [0, 360)
            preferences = np.arctan2(c2, c1) * (180/np.pi)
            preferences[preferences < 0] += 360
            preferences_rad = np.deg2rad(preferences)
            # Ensure preference lies within [0, 180) due to the period
            preferences = (preferences / 2) % 180
            # Calculate modulations
            modulations = 2 * (c2 - c1) / (np.sin(preferences_rad)
                                           - np.cos(preferences_rad))

        elif form == 'one_hot':
            preferences = 30 * np.argmax(tuning_coefs, axis=-1)
            modulations = \
                np.max(tuning_coefs, axis=-1) - np.min(tuning_coefs, axis=-1)

        else:
            raise ValueError('Form %s is not available.' % form)

        return modulations, preferences
