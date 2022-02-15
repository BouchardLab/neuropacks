import h5py
import numpy as np


class V1:
    """Processes V1 neurons obtained by the Ji lab.

    Parameters
    ----------
    data_path : string
        The path to the dataset.

    Attributes
    ----------
    data_path : string
        The location of the dataset.
    angles : np.ndarray, shape (n_samples,)
        The angle for each trial.
    unique_angles : np.ndarray
        The (sorted) unique angles.
    responses_by_stim : np.ndarray, shape (n_cells, n_samples, n_timestamps)
        The responses across cells, for each stimulus.
    n_cells : int
        The number of cells in the dataset.
    n_samples : int
        The number of samples in the dataset.
    n_trials : int
        The number of repetitions per angle.
    n_timestamps_stim : int
        The number of timestamps per stimulus window.
    n_angles : int
        The number of unique angles.
    """
    def __init__(self, data_path):
        self.data_path = data_path
        self._read_data()

    @staticmethod
    def _read_data_single_file(data_path):
        """Reads in retinal responses from the provided data."""
        with h5py.File(data_path, 'r') as data:
            # get angles per trials
            angles = np.squeeze(data['order'][0][::25])
            # get responses segmented by stimulus
            responses_by_stim = data['dff0'][:]
        responses_by_stim = responses_by_stim.reshape(responses_by_stim.shape[0],
                                                      -1,
                                                      25)[..., 5:]
        return angles, responses_by_stim

    def _read_data(self):
        """Reads in retinal responses from the provided data."""
        if isinstance(self.data_path, list):
            self.angles = []
            self.responses_by_stim = []
            for dp in self.data_path:
                results = V1._read_data_single_file(self.data_path)
                self.angles.append(results[0])
                self.responses_by_stim.append(results[1])
            self.angles = np.concatenate(self.angles)
            self.responses_by_stim = np.concatenate(self.respons)
        else:
            self.angles, self.responses_by_stim = V1._read_data_single_file(self.data_path)

        self.unique_angles = np.unique(self.angles)
        # dataset dimensions
        self.n_cells, self.n_samples, self.n_timestamps_stim = self.responses_by_stim.shape
        self.n_angles = self.unique_angles.size
        self.n_trials = int(self.n_samples / self.n_angles)

    def cells_to_idx(self, cells):
        """Converts cells input to indices.

        Parameters
        ----------
        cells : string or np.ndarray
            The cells to query. If 'all' (default), responses from all cells are
            used. If 'tuned', only the tuned cells are used. If np.ndarray,
            those indices are used directly.
        """
        if cells == 'all':
            cells = np.arange(self.n_cells)
        elif not isinstance(cells, np.ndarray):
            raise ValueError('Invalid type for cells.')
        return cells

    def get_design_matrix(self, form='angle', angles=None):
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
        if angles is None:
            angles = np.copy(self.angles)

        if form == 'angle':
            # the angles for each trial; no extra dimension required
            X = angles
        elif form == 'cosine':
            X = np.zeros((angles.size, 2))
            X[:, 0] = np.cos(np.deg2rad(angles))
            X[:, 1] = np.sin(np.deg2rad(angles))

        elif form == 'cosine2':
            X = np.zeros((angles.size, 2))
            X[:, 0] = np.cos(2 * np.deg2rad(angles))
            X[:, 1] = np.sin(2 * np.deg2rad(angles))

        else:
            raise ValueError("Incorrect design matrix form specified.")

        return X

    def get_response_matrix(self, cells='all', response='max'):
        """Obtains a response matrix from the cellular responses.

        Parameters
        ----------
        cells : string or np.ndarray
            The cells to query from the responses. If 'all' (default),
            responses from all cells are used. If 'tuned', only the tuned
            cells are used. If np.ndarray, those indices are used directly.
        response : string
            How to calculate the response using the flourescence across time.
            Default is 'max', where the maximum for each trial constitutes the
            response.

        Returns
        -------
        X : np.ndarray, shape (n_samples, n_cells)
            The design matrix.
        """
        cells = self.cells_to_idx(cells)

        # calculate response per trial
        if response == 'max':
            X = np.max(self.responses_by_stim[cells], axis=-1).T
        else:
            raise NotImplementedError('Other response types not implemented.')

        return X

    def get_mean_response_by_angle(self, cells='all', response='max'):
        """Get the mean response of cells, per unique angle.

        Parameters
        ----------
        cells : string or np.ndarray
            The cells to query from the responses. If 'all' (default),
            responses from all cells are used. If 'tuned', only the tuned
            cells are used. If np.ndarray, those indices are used directly.
        response : string
            How to calculate the response using the flourescence across time.
            Default is 'max', where the maximum for each trial constitutes the
            response.

        Returns
        -------
        X_tuned : np.ndarray, shape (n_angles, n_cells)
            The average response per unique angle, per cell.
        """
        # get responses for all queried cells
        X = self.get_response_matrix(cells, response)
        # get average response per unique angle
        X_trial_avg = np.zeros((self.n_angles, X.shape[1]))
        for idx, angle in enumerate(self.unique_angles):
            X_trial_avg[idx] = X[self.angles == angle].mean(axis=0)
        return X_trial_avg

    def get_tuning_curve(self, form, tuning_coefs, intercept=None, angles=None):
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
            angles = np.linspace(self.unique_angles[0],
                                 self.unique_angles[-1],
                                 1000)

        # Calculate design matrix for angles
        X = self.get_design_matrix(form=form, angles=angles)

        # Calculate tuning curve
        tuning_curve = np.dot(X, tuning_coefs)
        if intercept is not None:
            tuning_curve += intercept

        return angles, tuning_curve
