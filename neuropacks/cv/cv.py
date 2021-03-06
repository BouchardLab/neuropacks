import h5py
import numpy as np


class CV:
    def __init__(self, data_path):
        """Processes consonant-vowel perception dataset obtained from the
        Chang lab. This class operates on the processed neural data.

        Parameters
        ----------
        data_path : string
            The path to the dataset.
        """
        self.data_path = data_path
        # Import dataset
        with h5py.File(self.data_path, 'r') as data:
            self.X = data['X'][:]
            self.cvs = data['cvs'][:].astype('str')
            self.speakers = data['speakers'][:]
            self.cv_accuracy = data['cv_accuracy'][:]
            self.cv_accuracy_baseline = data['cv_accuracy_baseline'][:]
            self.c_accuracy = data['c_accuracy'][:]
            self.c_accuracy_baseline = data['c_accuracy_baseline'][:]
            self.v_accuracy = data['v_accuracy'][:]
            self.v_accuracy_baseline = data['v_accuracy_baseline'][:]

        # Get consonants and vowels
        self.cs = np.array([cv[:-1] for cv in self.cvs])
        self.vs = np.array([cv[-1] for cv in self.cvs])
        # Get unique CVs, consonants, and vowels
        self.unique_cvs = np.unique(self.cvs)
        self.unique_cs = np.unique(self.cs)
        self.unique_vs = np.unique(self.vs)
        # Get number of unique CVs, consonants, and vowels
        self.n_cvs = self.unique_cvs.size
        self.n_cs = self.unique_cs.size
        self.n_vs = self.unique_vs.size

    def get_design_matrix(self, stimulus='cv'):
        """Extracts the design matrix for this CV experiment. In this setting,
        its consists of a vector denoting which CV, consonant, or vowel served
        as the stimulus.

        Parameters
        ----------
        stimulus : string
            The stimulus to use: either 'cv', 'c', or 'v'.

        Returns
        -------
        design_matrix : np.ndarray
            The design matrix.
        """
        if stimulus == 'cv':
            return self.cvs
        elif stimulus == 'c':
            return self.cs
        elif stimulus == 'v':
            return self.vs
        else:
            raise ValueError("Incorrect stimulus input.")

    def get_response_matrix(self, stimulus='cv', norm=False):
        """Extracts the response matrix. Chooses the 'response' timepoint
        according to the stimulus.

        Parameters
        ----------
        stimulus : string
            The stimulus to use: either 'cv', 'c', or 'v'.
        norm : bool
            If True, get the timepoint according to the normalized maximum
            accuracy.

        Returns
        -------
        X : np.ndarray
            The response matrix.
        """
        # Identify the best response time for stimulus
        if stimulus == 'cv':
            metric = self.cv_accuracy
            baseline = self.cv_accuracy_baseline
        elif stimulus == 'c':
            metric = self.c_accuracy
            baseline = self.c_accuracy_baseline
        elif stimulus == 'v':
            metric = self.v_accuracy
            baseline = self.v_accuracy_baseline
        else:
            raise ValueError("Incorrect stimulus input.")
        # Get response matrix at specific timepoint
        if norm:
            timepoint = np.argmax(metric / baseline)
        else:
            timepoint = np.argmax(metric)
        X = self.X[:, timepoint, :]
        return X
