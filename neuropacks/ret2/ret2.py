import h5py
import numpy as np


class RET2:
    """Processes retinal ganglion cell recordings obtained by the Feller lab.

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

    responses : np.ndarray, shape (n_cells, n_timestamps)
        The responses across the entire timecourse for each cell.

    responses_by_stim : np.ndarray, shape (n_cells, n_samples, n_timestamps)
        The responses across cells, for each stimulus.

    timestamps : np.ndarray
        The timestamps for the recording session.

    tuned_cells : np.ndarray
        The indices for the tuned cells.

    n_cells : int
        The number of cells in the dataset.

    n_samples : int
        The number of samples in the dataset.

    n_trials : int
        The number of repetitions per angle.

    n_timestamps_stim : int
        The number of timestamps per stimulus window.

    n_timestamps : int
        The total number of timestamps over the recording session.

    n_angles : int
        The number of unique angles.
    """
    def __init__(self, data_path):
        self.data_path = data_path
        self._read_data()

    def _read_data(self):
        """Reads in retinal responses from the provided data."""
        with h5py.File(self.data_path, 'r') as data:
            # get sampling rate
            self.fs = data['fs'][:].item()
            # get angles per trials
            self.angles = np.squeeze(data['stimDirs'][:])
            # get all responses
            self.responses = data['dF'][:]
            # get responses segmented by stimulus
            self.responses_by_stim = data['stimDF'][:]
            # get tuned cells
            self.tuned_cells = np.squeeze(data['dsRois'])[:].astype('int') - 1

        self.unique_angles = np.unique(self.angles)
        # dataset dimensions
        self.n_cells, self.n_samples, self.n_timestamps_stim = self.responses_by_stim.shape
        self.n_angles = self.unique_angles.size
        self.n_trials = int(self.n_samples / self.n_angles)
        # get timestamps
        self.n_timestamps = self.responses.shape[1]
        self.timestamps = np.arange(self.n_timestamps) / self.fs

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
        elif cells == 'tuned':
            cells = self.tuned_cells
        elif not isinstance(cells, np.ndarray):
            raise ValueError('Invalid type for cells.')
        return cells

    def get_design_matrix(self, cells='all', response='max'):
        """Obtains a design matrix from the cellular responses.

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
        X = self.get_design_matrix(cells, response)
        # get average response per unique angle
        X_trial_avg = np.zeros((self.n_angles, X.shape[1]))
        for idx, angle in enumerate(self.unique_angles):
            X_trial_avg[idx] = X[self.angles == angle].mean(axis=0)
        return X_trial_avg
