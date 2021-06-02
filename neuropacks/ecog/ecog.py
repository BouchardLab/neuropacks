import h5py
import numpy as np


class ECOG:
    def __init__(self, data_path, grid_path=None):
        """Processes and provides design/response matrices for the primary
        auditory cortex ECoG dataset from the Bouchard Lab.

        Parameters
        ----------
        data_path : string
            The path to the ECoG dataset.
        grid_path : string, optional
            Path to the grid file.

        Attributes
        ----------
        data_path : string
            The path to the ECoG dataset.
        responses : nd-array
            A numpy array containing the responses of the electrodes to all
            stimuli and trials. The responses are already decomposed into
            their frequency bands. This array has shape
                (n_timepoints, n_trials, n_stim_amps, n_stim_freqs,
                n_electrodes, n_bands)
        n_timepoints : int
            The number of timepoints for each trial.
        n_trials : int
            The number of repetitions for each frequency, amplitude
            combination.
        n_stim_amps : int
            The number of unique stimulus amplitudes.
        n_stim_freqs : int
            The number of unique stimulus frequencies.
        n_electrodes : int
            The number of electrodes.
        n_bands : int
            The number of frequency bands in the wavelet decomposition.
        n_total_trials : int
            The number of total trials, including all repetitions, stimulus
            values, and amplitude values.
        freq_set : nd-array
            The unique frequencies used for each tone pip in the experiment.
        log_freq_set : nd-array
            The unique log-frequencies used for each tone pip in the
            experiment.
        amp_set : nd-array
            The unique amplitude attenuations used for each tone pip in the
            experiment.
        bands : nd-array of strings
            The names corresponding to each frequency band in the wavelet
            decomposition.
        """
        self.data_path = data_path
        # Read in data
        data = h5py.File(data_path, 'r')
        # Extract ECoG responses
        self.responses = data['RspM']
        # Get shape bounds
        (self.n_timepoints, self.n_trials, self.n_stim_amps, self.n_stim_freqs,
         self.n_electrodes, self.n_bands) = self.responses.shape
        # Set experiment parameters
        self.n_total_trials = \
            self.n_trials * self.n_stim_freqs * self.n_stim_amps
        self.freq_set = data['FrqSet'][:].ravel()
        self.log_freq_set = np.log(self.freq_set)
        self.amp_set = data['AmpSet'][:].ravel()
        self.bands = np.array(['B', 'G', 'HG', 'UHG', 'MUAR', 'tMUA'])
        # Read in grid
        self.read_grid(grid_path)
        # PAC electrodes
        self.pac_idxs = np.array([
            1, 8, 9, 16, 17, 18, 24, 25, 26, 28, 33, 35, 37, 39, 41, 42, 43, 44,
            45, 48, 49, 50, 51, 52, 53, 54, 55, 57, 74, 76, 77, 78, 79, 95, 97,
            98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
            112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124,
            125, 126, 127])

    def band_to_idx(self, band):
        """Converts a frequency band to the correct index in the data array.

        Parameters
        ----------
        band : string or int
            If a string, will convert to an index within self.bands. If a valid
            int, returns the same int.

        Returns
        -------
        idx : int
            The index of the provided frequency band.
        """
        # Band is a string
        if isinstance(band, str):
            idx = np.argwhere(self.bands == band)
            # Band was not found in self.bands
            if idx.size == 0:
                raise ValueError('%g is not a correct frequency band.' % band)
            else:
                idx = np.asscalar(np.argwhere(self.bands == band))
        # Band is an int
        elif isinstance(band, int):
            # Check that it's within the correct bounds
            if band < self.bands.size and band >= 0:
                idx = band
            else:
                raise ValueError('Index must be between 0 and %s, inclusive.'
                                 % self.bands - 1)
        else:
            raise ValueError('Band is not correctly specified.')

        return idx

    def get_peaks(self, bounds, electrodes=None):
        """Extract the peaks from the electrode responses.

        Parameters
        ----------
        bounds : tuple of two ints
            A tuple whose entries indicate the starting and ending indices
            which bound the region over which the peak will be taken.
        electrodes : array-like
            A list, int, or numpy array of indices denoting which electrodes
            to extract peaks from. If None, all electrodes are extracted.

        Returns
        -------
        peaks : nd-array
            The peaks for each response, taken over the specified bounds.
        """
        if electrodes is None:
            peaks = np.max(self.responses[bounds[0]:bounds[1], ...], axis=0)
        else:
            peaks = np.max(
                self.responses[bounds[0]:bounds[1], :, :, :, electrodes, :],
                axis=0)
        return peaks

    def get_responses(self, bounds, band, electrodes=None):
        """Get the peak responses for specified bounds and a frequency band.

        Parameters
        ----------
        bounds : tuple of two ints
            A tuple whose entries indicate the starting and ending indices
            which bound the region over which the peak will be taken.
        electrodes : array-like
            A list, int, or numpy array of indices denoting which electrodes
            to extract peaks from. If None, all electrodes are extracted.
        band : string or int
            A string or int indicating which frequency band to use.

        Returns
        -------
        responses_per_electrode : nd-array, shape
            (n_trials, n_stim_amps, n_freq_amps, n_electrodes)
            The peak responses for a given frequency band across all
            electrodes.
        """
        # Convert band to index if it's needed
        band = self.band_to_idx(band)
        # Get peak responses for time bounds and electrodes
        peaks = self.get_peaks(bounds=bounds, electrodes=electrodes)
        # Extract the responses for a given band
        responses_per_electrode = peaks[..., band]
        return responses_per_electrode

    def get_response_matrix(
        self, bounds, band, electrodes=None, amps=None, transform=None
    ):
        """Creates the response matrix.

        Parameters
        ----------
        bounds : tuple of two ints
            A tuple whose entries indicate the starting and ending indices
            which bound the region over which the peak will be taken.
        band : string or int
            A string or int indicating which frequency band to use.
        electrodes : array-like
            A list, int, or numpy array of indices denoting which electrodes
            to extract peaks from. If None, all electrodes are extracted.
        transform : string, optional
            The transform to apply to the responses. If None, no transform
            is applied.

        Returns
        -------
        Y : nd-array, shape (n_trials, n_electrodes)
            Response matrix.
        """
        # Convert band to index if it's needed
        band = self.band_to_idx(band)
        # Extract responses per electrode
        responses_per_electrode = self.get_responses(
            bounds=bounds,
            band=band,
            electrodes=electrodes)
        if amps is not None:
            responses_per_electrode = responses_per_electrode[:, amps, :, :]
            n_stim_amps = len(amps)
        else:
            n_stim_amps = self.n_stim_amps

        Y = responses_per_electrode.reshape(
            self.n_trials * n_stim_amps * self.n_stim_freqs, -1
        )
        # Apply desired transform
        if transform == 'sqrt':
            Y = np.sqrt(Y)
        return Y

    def get_design_matrix(
        self, form='bf', n_gaussians=7, lower_log_freq=None,
        upper_log_freq=None, var=0.64
    ):
        """Create the design matrix according to some specified form.

        Parameters
        ----------
        form : string
            The form of the design matrix. Possible values include:
                'bf' : Gaussian basis functions
                '1h' : One hot encoding of frequency
                'a1h' : Auditory one hot encoding of frequency
                'frequency' : Frequency value
                'id' : Unique frequency and amplitude ID
        n_gaussians : int, optional
            If design is 'bf', the number of gaussian basis functions.
        lower_log_freq : float, optional
            If design is 'bf', the lower frequency bound.
        upper_log_freq : float, optional
            If design is 'bf', the upper frequency bound.
        var : float, optional
            If design is 'bf', the variance of each Gaussian in octaves.

        Returns
        -------
        X : nd-array, shape (n_trials, n_features)
            Design matrix.
        """
        # Calculate number of trials
        n_total_trials = self.n_stim_freqs * self.n_stim_amps * self.n_trials

        if form == 'frequency' or form == 'id' or form == 'amplitude':
            # Initialize design matrix
            X = np.zeros(n_total_trials)

        elif form == '1h' or form == 'a1h':
            # Intialize design matrix
            X = np.zeros((n_total_trials, self.n_stim_freqs))

        elif form == 'bf' or form == 'abf':
            # Establish log frequency bounds
            if lower_log_freq is None:
                lower_log_freq = self.log_freq_set[1]
            if upper_log_freq is None:
                upper_log_freq = self.log_freq_set[-2]
            # Create parameters for radial basis functions
            means = np.linspace(lower_log_freq, upper_log_freq, n_gaussians)
            norm = 1./np.sqrt(2 * np.pi * var)
            # Intialize design matrix
            X = np.zeros((n_total_trials, n_gaussians))

        else:
            raise ValueError('incorrect design matrix specified.')

        # Iterate over all trials
        for trial_idx in range(self.n_trials):
            for amp_idx in range(self.n_stim_amps):
                for freq_idx in range(self.n_stim_freqs):
                    # Calculate index of current trial in design matrix
                    index = trial_idx * self.n_stim_amps * self.n_stim_freqs \
                            + amp_idx * self.n_stim_freqs \
                            + freq_idx
                    # Populate this trial with design matrix information
                    if form == 'frequency':
                        X[index] = self.freq_set[freq_idx]
                    elif form == 'amplitude':
                        X[index] = self.amp_set[amp_idx]
                    elif form == 'id':
                        X[index] = 100 * amp_idx + freq_idx
                    elif form == '1h':
                        X[index, freq_idx] = 1
                    elif form == 'a1h':
                        X[index, freq_idx] = 8 + self.amp_set[amp_idx]
                    elif form == 'bf':
                        # Extract stimulus information
                        log_freq = self.log_freq_set[freq_idx]
                        parametric_stim = norm * np.exp(
                            -(log_freq - means)**2/(2 * var)
                        )
                        X[index, :] = parametric_stim
                    elif form == 'abf':
                        # Extract stimulus information
                        log_freq = self.log_freq_set[freq_idx]
                        amplitude = self.amp_set[amp_idx]
                        parametric_stim = norm * np.exp(
                            -(log_freq - means)**2/(2 * var)
                        )
                        X[index, :] = amplitude * parametric_stim

        return X

    def get_tuning_curve(
        self, tuning_coefs, frequencies=None, lower_log_freq=None,
        upper_log_freq=None, var=0.64
    ):
        """Creates a tuning curve from basis function coefficients.

        Parameters
        ----------
        tuning_coefs : np.ndarray, shape (n_tuning_curves, n_gaussians)
            The tuning coefficients for each tuning curve.
        frequencies : nd-array
            The frequencies for which to calculate tuning curve values.
        lower_log_freq : float, optional
            Lower log frequency bound for tuning coefficients. If None,
            the lowest log frequency is used.
        upper_log_freq : float, optional
            Upper log frequency bound for tuning coefficients. If None,
            the highest log frequency is used.
        var : float, optional
            The variance of each Gaussian in octaves.

        Returns
        -------
        frequencies : np.ndarray
            Returns frequencies if it is provided. If frequencies was None,
            returns a spread across the default log-frequency space.
        tuning_curves : np.ndarray, same shape as frequencies
            The value of the tuning curve corresponding to each frequency in
            the first return argument.
        """
        if tuning_coefs.ndim == 1:
            tuning_coefs = tuning_coefs[np.newaxis]
        # Number of parameters
        n_curves, n_gaussians = tuning_coefs.shape

        # Establish log frequency bounds
        if lower_log_freq is None:
            lower_log_freq = self.log_freq_set[1]
        if upper_log_freq is None:
            upper_log_freq = self.log_freq_set[-2]
        # Create parameters for radial basis functions
        means = np.linspace(lower_log_freq, upper_log_freq, n_gaussians)
        norm = 1./np.sqrt(2 * np.pi * var)

        # Frequencies for tuning curve
        if frequencies is None:
            frequencies = np.linspace(self.freq_set[0], self.freq_set[-1], 1000)
        log_freqs = np.log(frequencies)
        # Calculate outputs of tuning curves
        tuning_curves = np.zeros((n_curves, frequencies.size))
        for idx, tuning_coef in enumerate(tuning_coefs):
            tuning_curves[idx] = np.sum(
                tuning_coef * norm * np.exp(
                    -np.subtract.outer(log_freqs, means)**2 / (2 * var)
                ), axis=1)
        return frequencies, tuning_curves

    def get_tuning_modulation_and_preference(self, tuning_coefs, form='bf'):
        """Calculates tuning modulation given a set of tuning coefficients.

        Parameters
        ----------
        tuning_coefs : np.ndarray, shape (n_tuning_curves, n_features)
            The tuning coefficients.
        form : string
            The form of the tuning coefficients. Defaults to basis functions.

        Returns
        -------
        modulations : np.ndarray
            Tuning modulations (min-to-max distance)
        preferences : np.ndarray
            Tuning preferences.
        """
        if tuning_coefs.ndim == 1:
            tuning_coefs = tuning_coefs[np.newaxis]

        # Calculate the tuning curve for basis functions
        if form == 'bf':
            frequencies, tuning_curves = self.get_tuning_curve(tuning_coefs)
            # Tuning modulation
            modulations = np.max(tuning_curves, axis=1) - np.min(tuning_curves, axis=1)
            # Tuning preference
            preference_idx = np.argmax(tuning_curves, axis=1).ravel()
            preferences = frequencies[preference_idx]
        # Otherwise, we can calculate the modulation directly
        elif form == '1h' or form == 'a1h':
            modulations = np.max(tuning_coefs, axis=1) - np.min(tuning_coefs, axis=1)
            preference_idx = np.argmax(tuning_coefs, axis=1).ravel()
            preferences = self.freq_set[preference_idx]
        else:
            raise ValueError('Incorrect tuning specified.')

        return modulations, preferences

    def read_grid(self, grid_path):
        """Extract the grid from its path. Helper function for initializing the
        class.

        Parameters
        ----------
        grid_path : string
            Path to the grid file.
        """
        # Grid path does not need to be provided.
        if grid_path is None:
            self.grid_path = None
            self.grid = None
        else:
            grid = h5py.File(grid_path, 'r')
            self.grid = grid['grdid'][:]
            grid.close()

    def get_xy_for_electrode(self, idx):
        """Gets the x,y coordinates for a provided electrode index.

        Parameters
        ----------
        idx : int
            The electrode index.

        Returns
        -------
        x, y : int
            The spatial x, y coordinates of the electrode.
        """
        # Check if grid exists
        if self.grid is None:
            raise ValueError('Grid is not set.')
        else:
            x, y = self.grid[:, idx]
            return int(x - 1), int(y - 1)
