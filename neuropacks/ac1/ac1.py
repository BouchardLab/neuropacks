import h5py
import numpy as np


class AC1:
    def __init__(self, data_path):
        """Processes and provides design/response matrices for the
        micro-electrocorticography recordings auditory cortex recordings from
        the Bouchard lab.

        Parameters
        ----------
        data_path : string
            The path to the dataset.
        amplitudes : int, array-like, or None
            The amplitudes, as indices, to use in the responses. If None,
            all are used.

        Attributes
        ----------
        data_path : string
            The path to the dataset.
        responses : np.ndarray
            A numpy array containing the responses of the electrodes to all
            stimuli and trials. The responses are decomposed into the dimensions
            (n_units, n_trials, n_frequencies, n_amplitudes)
        n_units : int
            The number of electrodes.
        n_trials : int
            The number of repetitions for each frequency, amplitude
            combination.
        n_frequencies : int
            The number of unique frequencies.
        n_amplitudes : int
            The number of amplitudes.
        """
        self.data_path = data_path
        self._read_data()

    def _read_data(self):
        """Reads in the electrocorticography responses."""
        with h5py.File(self.data_path, 'r') as data:
            self.frequencies = data.attrs['frequencies']
            self.responses = data['final_rsp'][:]

        self.n_units, self.n_trials, self.n_frequencies, self.n_amplitudes = \
            self.responses.shape

    def check_amplitudes(self, amplitudes=None):
        """Converts a set of amplitudes to proper indices.

        Parameters
        ----------
        amplitudes : int, array-like, or None
            The amplitudes, as indices, to use in the responses. If None,
            all are used.

        Returns
        -------
        amplitudes : np.ndarray
            The amplitude indices.
        """
        if amplitudes is None:
            amplitudes = np.arange(self.n_amplitudes)
        elif isinstance(amplitudes, int):
            amplitudes = np.array([amplitudes])
        elif isinstance(amplitudes, list):
            amplitudes = np.array(amplitudes)
        elif not isinstance(amplitudes, np.ndarray):
            raise ValueError('Amplitudes should be int, array-like, or None.')
        return amplitudes

    def get_design_matrix(self, form='stimuli', amplitudes=None):
        """Extracts the design matrix for a set of amplitudes.

        Parameters
        ----------
        form : string
            The form of the design matrix. Currently, only 'stimuli' is
            implented.
        amplitudes : int, array-like, or None
            The amplitudes, as indices, to use in the responses. If None,
            all are used.

        Returns
        -------
        X : np.ndarray
            The design matrix.
        """
        amplitudes = self.check_amplitudes(amplitudes)
        n_amplitudes = amplitudes.size
        frequencies = np.repeat(self.frequencies, n_amplitudes * self.n_trials)

        if form == 'stimuli':
            X = np.copy(frequencies)
        else:
            raise NotImplementedError('Other design matrix forms not implemented.')

        return X

    def get_response_matrix(self, amplitudes=None):
        """Extracts the response matrix for a set of amplitudes.

        Parameters
        ----------
        amplitudes : int, array-like, or None
            The amplitudes, as indices, to use in the responses. If None,
            all are used.

        Returns
        -------
        X : np.ndarray, shape (n_samples, n_units)
            The design matrix, with frequencies and amplitudes collapsed into
            a single sample dimension.
        """
        amplitudes = self.check_amplitudes(amplitudes)
        responses = self.responses[..., amplitudes]
        X = np.transpose(responses, [2, 3, 1, 0]).reshape(-1, self.n_units)
        return X
