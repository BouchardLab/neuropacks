import numpy as np
from scipy.io import loadmat


class PVC11():
    def __init__(self, data_path):
        self.data_path = data_path
        self.load_data()
        self.get_spike_counts()

    def load_data(self):
        """Loads the PVC-11 data."""

        # load data
        data = loadmat(self.data_path, struct_as_record=False)
        # extract events
        self.events = data['data'][0, 0].EVENTS
        # extract experiment parameters
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

        # iterate over all trials, stimuli, and neurons
        for neuron in range(self.n_neurons):
            for stim in range(self.n_stimuli):
                for trial in range(self.n_trials):
                    # compute all spike counts in the trial
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

    def get_design_matrix(self, form='angle'):
        """Create design matrix according to a specified form.

        Parameters
        ----------
        form : string
            The structure of the design matrix.

        Returns
        -------
        X : nd-array, shape (n_trials, n_features)
            The design matrix.
        """

        unique_angles = self.get_angles()
        angles = np.repeat(unique_angles, self.n_trials)
        if form == 'angle':
            # the angles for each trial; no extra dimension required
            X = angles

        elif form == 'label':
            X = (angles/30).astype('int64')

        elif form == 'cosine':
            X = np.zeros((angles.size, 2))
            X[:, 0] = np.cos(np.deg2rad(angles))
            X[:, 1] = np.sin(np.deg2rad(angles))

        elif form == 'gaussian':
            X = np.zeros((angles.size, 2))
            X[:, 0] = np.deg2rad(angles)
            X[:, 1] = np.deg2rad(angles)**2

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

        # apply any desired transform
        if transform == 'square_root':
            Y = np.sqrt(Y)
        elif transform == 'log':
            Y = np.log(Y)
        else:
            if transform is not None:
                raise ValueError("Transform %s is not recognized." % transform)

        return Y
