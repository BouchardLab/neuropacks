import numpy as np
from scipy.io import loadmat

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.metrics import r2_score

from pyuoi import UoI_Lasso


class RET1:
    def __init__(self, data_path, random_path):
        self.data_path = data_path
        self.random_path = random_path

        # load neural response data
        data = loadmat(self.data_path, struct_as_record=False)

        self.recordings = data['stimulus'].ravel()
        self.n_recordings = self.recordings.size
        self.spikes = data['spikes']
        self.n_cells = np.asscalar(
            np.asscalar(data['datainfo']).Ncell
        )

    def check_recording_idx(self, recording_idx):
        """Check to see if recording index is correctly specified.

        Parameters
        ----------
        recording_idx : int
            The recording index.
        """

        if not isinstance(recording_idx, int):
            raise ValueError('Recording index must be an integer.')

        if recording_idx < 0 or recording_idx >= self.n_recordings:
            raise IndexError('Recording index outside bounds.')

        return

    def check_cells(self, cells):
        """Check 'cells' input, and potentially convert to numpy array,
        if necessary.

        Parameters
        ----------
        cells : int, list, np.ndarray or None
            The set of cell indices under consideration. If an int or list,
            converts to an ndarray. If None, creates a numpy array of all
            cells.

        Returns
        -------
        cells : np.ndarray
            The converted numpy array containing the cell indices.
        """

        # None corresponds to using all cells
        if cells is None:
            cells = np.arange(self.n_cells)
        # only one cell
        elif isinstance(cells, int):
            cells = np.array([cells])
        # multiple cells, but convert to numpy array
        elif isinstance(cells, list):
            cells = np.array(cells)
        elif not isinstance(cells, np.ndarray):
            raise ValueError('cells variable type is not correct.')

        return cells

    def get_n_frames_per_window(self, recording_idx, window_length=0.5):
        """Returns the number of frames in a specified recording and
        window size.

        Parameters
        ----------
        recording_idx : int
            The recording index.

        window_length : float
            The length of window for which to calculate the STRF, in seconds.

        Returns
        -------
        n_frames_per_window : float
            The number of stimulus frames per window length.
        """

        # extract recording
        self.check_recording_idx(recording_idx=recording_idx)
        recording = self.recordings[recording_idx]

        # extract frame length
        frame_length = np.asscalar(recording.frame)

        # calculate number of frames per window, given specified size
        n_frames_per_window = int(np.round(
            window_length / frame_length
        ))

        return n_frames_per_window

    def get_timestamps_for_recording(self, recording_idx):
        """Get timestamps for specified recording.

        Parameters
        ----------
        recording_idx : int
            The recording index.

        Returns
        -------
        timestamps : np.ndarray
            The array of timestamps for each recording.
        """

        # extract current recording
        self.check_recording_idx(recording_idx=recording_idx)
        recording = self.recordings[recording_idx]

        # get the number of samples by creating timestamps
        n_frames = np.asscalar(recording.Nframes)
        frame_length = np.asscalar(recording.frame)
        onset = np.asscalar(recording.onset)

        # calculate timestamps and number of samples
        timestamps = np.arange(n_frames) * frame_length + onset

        return timestamps

    def get_n_features_for_recording(self, recording_idx, window_length=0.5):
        """Get number of features (distinct bar dimensions) for recording.

        Parameters
        ----------
        recording_idx : int
            The recording index.

        window_length : float
            The length of window for which to calculate STRF, in seconds.

        Returns
        -------
        n_features : int
            The number of stimulus features.
        """

        # extract current recording
        self.check_recording_idx(recording_idx=recording_idx)
        recording = self.recordings[recording_idx]

        # extract parameter information
        params = np.asscalar(recording.param)
        # number of x/y dimensions
        Nx = np.asscalar(params.x) / np.asscalar(params.dx)
        Ny = np.asscalar(params.y) / np.asscalar(params.dy)
        # calculate number of features
        n_features = int(Nx * Ny)

        return n_features

    def get_dimensions_for_recording(self, recording_idx, window_length=0.5):
        """Gets both the number of features and samples for specified recording
        and window length.

        Parameters
        ----------
        recording_idx : int
            The recording index.

        window_length : float
            The length of window for which to calculate STRF, in seconds.

        Returns
        -------
        n_features : int
            The number of stimulus features.

        n_samples : int
            The number of samples in dataset.

        timestamps : np.ndarray
            The timestamps for dataset.
        """

        # get timestamps
        timestamps = self.get_timestamps_for_recording(
            recording_idx=recording_idx
        )
        n_samples = timestamps.size - 1

        # get number of features
        n_features = self.get_n_features_for_recording(
            recording_idx=recording_idx,
            window_length=window_length
        )

        return n_features, n_samples, timestamps

    def get_responses_for_recording(
        self, recording_idx, window_length=0.5, cells=None
    ):
        """Create response matrix for specified recording, window length, and
        choice of cells.

        Parameters
        ----------
        recording_idx : int
            The recording index.

        window_length : float
            The length of window for which to calculate STRF, in seconds.

        cells : int, list, np.ndarray or None
            The set of cell indices under consideration. If an int or list,
            converts to an ndarray. If None, creates a numpy array of all
            cells.

        Returns
        -------
        responses : np.ndarray, shape (n_samples, n_cells)
            A numpy array containing the responses for each cell in
            the recording.
        """

        # convert cells to numpy array, if necessary
        cells = self.check_cells(cells=cells)

        # set up responses matrix
        n_features, n_samples, timestamps = self.get_dimensions_for_recording(
            recording_idx=recording_idx,
            window_length=window_length
        )
        n_cells = cells.size
        responses = np.zeros((n_samples, n_cells))

        n_frames_per_window = self.get_n_frames_per_window(
            recording_idx=recording_idx,
            window_length=window_length
        )

        # iterate over cells, grabbing responses
        for idx, cell in enumerate(cells):
            # extract spike times
            spike_times = self.spikes[cell, recording_idx]

            # binning
            binned_spikes, _ = np.histogram(spike_times, bins=timestamps)
            # zero out the delay
            binned_spikes[:n_frames_per_window - 1] = 0

            # put binned spike counts in response matrix
            n_spikes = np.sum(binned_spikes)
            responses[:, idx] = binned_spikes / n_spikes

        return responses

    def get_stims_for_recording(self, recording_idx, window_length=0.5):
        """Create response matrix for specified recording and window length.

        Parameters
        ----------
        recording_idx : int
            recording index

        window_length : float
            length of window for which to calculate STRF, in seconds

        Returns
        -------
        stimuli : np.ndarray, shape (n_features, n_samples)
            A numpy array containing white noise stimuli shown to the cells.
        """

        # get dimensions
        n_features, n_samples, _ = self.get_dimensions_for_recording(
            recording_idx=recording_idx,
            window_length=window_length
        )

        # read in the random bits as bytes
        byte = np.fromfile(
            self.random_path,
            count=n_samples * n_features // 8,
            dtype='uint8'
        )
        # convert to bits
        stimuli = np.unpackbits(byte).astype('float32')
        # convert bits to +1/-1
        stimuli = 2 * stimuli - 1

        stimuli = stimuli.reshape((n_samples, n_features)).T

        return stimuli

    def calculate_strf_for_neurons(
        self, method, recording_idx, window_length=0.5, cells=None,
        test_frac=None, return_scores=False, verbose=False, **kwargs
    ):
        """Calculates the STRFs for specified neurons and a specified method.

        Parameters
        ----------
        method : string
            The regression method to use when calculating STRFs.

        recording_idx : int
            The recording index to obtain design and response matrices.

        window_length : float
            The number of seconds to fit in STRF window.

        test_frac : float or None
            The fraction of data to use as a test set. If None, the entire set
            will be used only for training.

        return_scores : bool
            A flag indicating whether to return explained variance over window.

        cells : int, list, np.ndarray or None
            The set of cell indices under consideration. If None, creates a
            numpy array of all cells.

        verbose : bool
            If True, function will output which frame it is currently fitting.

        Returns
        -------
        strf : np.ndarray, shape (n_cells, n_frames_per_window, n_features)
            A numpy array containing the spatio-temporal receptive field.

        intercepts : np.ndarray, shape (n_cells, n_frames_per_window)
            A numpy array containing the intercepts for the STRFs.

        training_scores : tuple of np.ndarrays, each with shape
                            (n_cells, n_frames_per_window)
            A tuple of numpy arrays containing scores measuring the predictive
            power of the STRF for each frame in the window. Returned only if
            requested.

        test_scores : tuple of np.ndarrays, each with shape
                            (n_cells, n_frames_per_window)
            A tuple of numpy arrays containing scores measuring the predictive
            power of the STRF, but on a test set for each frame in the window.
            Returned only if requested. If the test fraction is None,
            test_scores will be returned as None.
        """

        # set up array of cells to iterate over
        cells = self.check_cells(cells=cells)

        # extract design and response matrices
        stimuli = self.get_stims_for_recording(
            recording_idx=recording_idx,
            window_length=window_length
        )
        responses = self.get_responses_for_recording(
            recording_idx=recording_idx,
            window_length=window_length,
            cells=cells
        )
        # number of frames that will appear in window length
        n_frames_per_window = self.get_n_frames_per_window(
            recording_idx=recording_idx,
            window_length=window_length
        )

        # create object to perform fitting
        if method == 'OLS':
            fitter = LinearRegression()

        elif method == 'Ridge':
            fitter = RidgeCV(
                cv=kwargs.get('cv', 5)
            )

        elif method == 'Lasso':
            fitter = LassoCV(
                normalize=kwargs.get('normalize', True),
                cv=kwargs.get('cv', 5),
                max_iter=kwargs.get('max_iter', 10000)
            )

        elif method == 'UoI_Lasso':
            fitter = UoI_Lasso(
                normalize=kwargs.get('normalize', True),
                n_boots_sel=kwargs.get('n_boots_sel', 30),
                n_boots_est=kwargs.get('n_boots_est', 30),
                selection_frac=kwargs.get('selection_frac', 0.8),
                estimation_frac=kwargs.get('estimation_frac', 0.8),
                n_lambdas=kwargs.get('n_lambdas', 30),
                stability_selection=kwargs.get('stability_selection', 1.),
                estimation_score=kwargs.get('estimation_score', 'BIC')
            )

        else:
            raise ValueError('Method %g is not available.' % method)

        # extract dimensions and create storage
        n_features, n_samples = stimuli.shape
        n_cells = cells.size
        strf = np.zeros((n_cells, n_frames_per_window, n_features))
        intercepts = np.zeros((n_cells, n_frames_per_window))

        # training and test score storage
        r2s_training = np.zeros((n_cells, n_frames_per_window))
        aics_training = np.zeros((n_cells, n_frames_per_window))
        bics_training = np.zeros((n_cells, n_frames_per_window))

        # if we evaluate on a test set, split up the data
        if test_frac is not None:
            n_test_samples = int(test_frac * n_samples)

            # split up stimulus
            # the samples axis is different for the stimuli and responses
            # matrices
            stimuli_test, stimuli = np.split(
                stimuli, [n_test_samples], axis=1
            )
            responses_test, responses = np.split(
                responses, [n_test_samples], axis=0
            )
            r2s_test = np.zeros((n_cells, n_frames_per_window))
            aics_test = np.zeros((n_cells, n_frames_per_window))
            bics_test = np.zeros((n_cells, n_frames_per_window))

        # iterate over cells
        for cell_idx, cell in enumerate(cells):
            if verbose:
                print('Cell ', cell)
            # copy response matrix
            responses_copy = np.copy(responses)
            if test_frac is not None:
                responses_test_copy = np.copy(responses_test)

            # iterate over frames in window
            for frame in range(n_frames_per_window):
                if verbose:
                    print('  Frame ', frame)
                # perform fit
                fitter.fit(stimuli.T, responses_copy[:, cell_idx])
                # extract coefficients
                strf[cell_idx, frame, :] = fitter.coef_.T
                intercepts[cell_idx, frame] = fitter.intercept_

                # scores
                y_true = responses_copy[:, cell_idx]
                y_pred = fitter.intercept_ + np.dot(stimuli.T, fitter.coef_)
                n_features = np.count_nonzero(fitter.coef_) + 1

                # explained variance
                r2s_training[cell_idx, frame] = r2_score(
                    y_true=y_true,
                    y_pred=y_pred
                )

                # bics
                bics_training[cell_idx, frame] = self.BIC(
                    y_true=y_true,
                    y_pred=y_pred,
                    n_features=n_features
                )

                # aics
                aics_training[cell_idx, frame] = self.AIC(
                    y_true=y_true,
                    y_pred=y_pred,
                    n_features=n_features
                )

                # roll the window up
                responses_copy = np.roll(responses_copy, -1, axis=0)

                # act on test set if necessary
                if test_frac is not None:
                    y_true_test = responses_test_copy[:, cell_idx]
                    y_pred_test = fitter.intercept_ \
                        + np.dot(stimuli_test.T, fitter.coef_)

                    # explained variance
                    r2s_test[cell_idx, frame] = r2_score(
                        y_true=y_true_test,
                        y_pred=y_pred_test
                    )

                    # bics
                    bics_test[cell_idx, frame] = self.BIC(
                        y_true=y_true_test,
                        y_pred=y_pred_test,
                        n_features=n_features
                    )

                    # aics
                    aics_test[cell_idx, frame] = self.AIC(
                        y_true=y_true_test,
                        y_pred=y_pred_test,
                        n_features=n_features
                    )

                    # roll the window up
                    responses_test_copy = np.roll(
                        responses_test_copy, -1, axis=0
                    )

        # get rid of potential unnecessary dimensions
        strf = np.squeeze(strf)
        r2s_training = np.squeeze(r2s_training)
        bics_training = np.squeeze(bics_training)
        aics_training = np.squeeze(aics_training)
        training_scores = (r2s_training, bics_training, aics_training)

        if test_frac is not None:
            r2s_test = np.squeeze(r2s_test)
            bics_test = np.squeeze(bics_test)
            aics_test = np.squeeze(aics_test)
            test_scores = (r2s_test, bics_test, aics_test)
        else:
            test_scores = None

        if return_scores:
            return strf, intercepts, training_scores, test_scores
        else:
            return strf, intercepts

    @staticmethod
    def BIC(y_true, y_pred, n_features):
        """Calculate the Bayesian Information Criterion under the assumption of
        normally distributed disturbances (which allows the BIC to take on the
        simple form below).

        Parameters
        ----------
        y_true : np.ndarray
            Array of true response values.

        y_pred : np.ndarray
            Array of predicted response values.

        n_features : int
            Number of features used in the model.

        Returns
        -------
        BIC : float
            The Bayesian Information Criterion.
        """

        n_samples = y_true.size

        # calculate residual sum of squares
        rss = np.sum((y_true - y_pred)**2)
        BIC = n_samples * np.log(rss / n_samples) \
            + n_features * np.log(n_samples)
        return BIC

    @staticmethod
    def AIC(y_true, y_pred, n_features):
        """Calculate the Akaike Information Criterion under the assumption of
        normally distributed disturbances. Utilizes a softer penalty on the
        model parsimony than the BIC.

        Parameters
        ----------
        y_true : np.ndarray
            Array of true response values.

        y_pred : np.ndarray
            Array of predicted response values.

        n_features : int
            Number of features used in the model.

        Returns
        -------
        AIC : float
            The Akaike Information Criterion.
        """

        n_samples = y_true.size

        # calculate residual sum of squares
        rss = np.sum((y_true - y_pred)**2)
        AIC = n_samples * np.log(rss / n_samples) \
            + n_features * 2
        return AIC

    @staticmethod
    def AICc(y_true, y_pred, n_features):
        """Calculate the corrected Akaike Information Criterion under the
        assumption of normally distributed disturbances. Modifies the parsimony
        penalty. Useful in cases when the number of samples is small.

        Parameters
        ----------
        y_true : np.ndarray
            Array of true response values

        y_pred : np.ndarray
            Array of predicted response values.

        n_features : int
            The number of features used in the model.

        Returns
        -------
        AICc : float
            The corrected Akaike Information Criterion.
        """

        n_samples = y_true.size

        # calculate residual sum of squares
        rss = np.sum((y_true - y_pred)**2)
        AICc = n_samples * np.log(rss / n_samples) \
            + n_features * 2 \
            + 2 * (n_features**2 + n_features) / (n_samples - n_features - 1)
        return AICc
