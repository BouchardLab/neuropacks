import h5py
import numpy as np
from scipy.io import loadmat
import time

from sklearn.linear_model import LinearRegression 
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.metrics import r2_score

from UoI_Lasso import UoI_Lasso

class Retina:
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
		'''Check to see if recording index is correctly specified.

		Parameters
		----------
		recording_idx : int
			recording index
		'''

		if not isinstance(recording_idx, int):
			raise ValueError('Recording index must be an integer.')

		if recording_idx < 0 or recording_idx >= self.n_recordings:
			raise IndexError('Recording index outside bounds.')

		return 

	def check_cells(self, cells):
		'''Check 'cells' input, and potentially convert to numpy array
		if necessary.

		Parameters
		----------
		cells : int, list, np.ndarray or None
			the set of cell indices under consideration. if the first two, converts
			to a np.ndarray. if None, creates a np array of all cells.

		Returns
		-------
		cells : np.ndarray
			the converted numpy array containing the cell indices
		'''

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
		'''Returns the number of frames in a specified recording and 
		window size.

		Parameters
		----------
		recording_idx : int
			recording index

		window_length : float
			length of window for which to calculate STRF, in seconds 
		
		Returns
		-------
		n_frames_per_window : float
			the number of stimuli frames per window length
		'''

		# extract recording
		self.check_recording_idx(recording_idx=recording_idx)
		recording = self.recordings[recording_idx]

		# extract frame length
		frame_length = np.asscalar(recording.frame)

		# calculate number of frames per window, given specified size
		n_frames_per_window = int(np.round(
			window_length/frame_length
		))

		return n_frames_per_window

	def get_timestamps_for_recording(self, recording_idx):
		'''Get timestamps for specified recording.

		Parameters
		----------
		recording_idx : int
			recording index

		Returns
		-------
		timestamps : np.ndarray
			array of timestamps for each recording
		'''

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
		'''Get number of features (distinct bar dimensions) for recording.

		Parameters
		----------
		recording_idx : int
			recording index

		window_length : float
			length of window for which to calculate STRF, in seconds 

		Returns
		-------
		n_features : int
			number of stimulus features
		'''

		# extract current recording
		self.check_recording_idx(recording_idx=recording_idx)
		recording = self.recordings[recording_idx]

		# extract parameter information
		params = np.asscalar(recording.param)
		# number of x/y dimensions
		Nx = np.asscalar(params.x)/np.asscalar(params.dx)
		Ny = np.asscalar(params.y)/np.asscalar(params.dy)
		# calculate number of features
		n_features = int(Nx * Ny)

		return n_features

	def get_dimensions_for_recording(self, recording_idx, window_length=0.5):
		'''Gets both the number of features and samples for specified recording
		and window length.

		Parameters
		----------
		recording_idx : int
			recording index

		window_length : float
			length of window for which to calculate STRF, in seconds 

		Returns
		-------
		n_features : int
			number of stimulus features

		n_samples : int
			number of samples in dataset

		timestamps : np.ndarray
			timestamps for dataset
		'''

		# get timestamps
		timestamps = self.get_timestamps_for_recording(recording_idx=recording_idx)
		n_samples = timestamps.size - 1

		# get number of features
		n_features = self.get_n_features_for_recording(
			recording_idx=recording_idx, 
			window_length=window_length
		)

		return n_features, n_samples, timestamps

	def get_responses_for_recording(self, recording_idx, window_length=0.5, cells=None):
		'''Create response matrix for specified recording, window length, and
		choice of cells.

		Parameters
		----------
		recording_idx : int
			recording index

		window_length : float
			length of window for which to calculate STRF, in seconds 

		cells : int, list, np.ndarray or None
			the set of cell indices under consideration. if the first two, converts
			to a np.ndarray. if None, creates a np array of all cells.

		Returns
		-------
		responses : np.ndarray 
			n_samples x n_cells array containing the responses for each cell
			in the recording.
		'''

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
			binned_spikes[:n_frames_per_window-1] = 0

			# put binned spike counts in response matrix
			n_spikes = np.sum(binned_spikes)
			responses[:, idx] = binned_spikes/n_spikes

		return responses

	def get_stims_for_recording(self, recording_idx, window_length=0.5):
		'''Create response matrix for specified recording and window length.

		Parameters
		----------
		recording_idx : int
			recording index

		window_length : float
			length of window for which to calculate STRF, in seconds 

		Returns
		-------
		stimuli : np.ndarray 
			n_features x n_samples array containing white noise stimuli
			shown to the cells.
		'''

		# get dimensions
		n_features, n_samples, _ = self.get_dimensions_for_recording(
			recording_idx=recording_idx, 
			window_length=window_length
		)

		# read in the random bits as bytes
		byte = np.fromfile(
			self.random_path, 
			count=n_samples*n_features//8, 
			dtype='uint8'
		)
		# convert to bits
		stimuli = np.unpackbits(byte).astype('float32')
		# convert bits to +1/-1
		stimuli = 2 * stimuli - 1

		stimuli = stimuli.reshape((n_samples, n_features)).T

		return stimuli

	def calculate_strf_for_neurons_and_frame(self, 
		method, recording_idx, frame, window_length=0.5, 
		return_score=False, cells=None, **kwargs
	):
		'''Calculates STRF for specified neurons.

		Parameters
		----------
		method : string
			regression method to use when calculating STRFs

		recording_idx : int
			recording index to obtain design and response matrices

		window_length : float
			number of seconds to fit in STRF window

		return_scores : bool
			flag indicating whether to return explained variance over window.

		cells : int, list, np.ndarray or None
			the set of cell indices under consideration. if the first two, converts
			to a np.ndarray. if None, creates a np array of all cells.

		Returns
		-------
		strf : np.ndarray
			n_cells x n_frames_per_window x n_features array describing the
			spatio-temporal receptive field.

		intercepts : np.ndarray
			n_cells x n_frames_per_window array containing the intercepts for
			the STRFs.

		r2s : np.ndarray
			returned only if requested; n_cells x n_frames_per_window array
			containing the explained variance of the STRFfor each frame 
			in the window.
		'''
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
		elif method == 'UoILasso':
			fitter = UoI_Lasso(
				normalize=kwargs.get('normalize', True),
				estimation_score=kwargs.get('estimation_score', 'BIC'),
				n_lambdas=kwargs.get('n_lambdas', 30),
				n_boots_sel=kwargs.get('n_boots_sel', 30),
				n_boots_est=kwargs.get('n_boots_est', 30),
				selection_thres_min=kwargs.get('selection_thres_min', 1.0),
			)
		else:
			raise ValueError('method %g is not available.' %method)

		# extract dimensions and create storage
		n_features, n_samples = stimuli.shape 
		n_cells = cells.size
		strf = np.zeros((n_cells, n_features))
		intercept = np.zeros(n_cells)
		r2 = np.zeros(n_cells)
		bic = np.zeros(n_cells)

		# iterate over cells
		for cell_idx, cell in enumerate(cells):
			# copy response matrix
			responses = np.roll(responses, -frame, axis=0)

			# perform fit
			fitter.fit(stimuli.T, responses[:, cell_idx])
			# extract coefficients
			strf[cell_idx, :] = fitter.coef_.T
			intercept[cell_idx] = fitter.intercept_

			# scores
			r2[cell_idx] = r2_score(
				responses[:, cell_idx], 
				fitter.intercept_ + np.dot(stimuli.T, fitter.coef_)
			)

		# get rid of potential unnecessary dimensions
		strf = np.squeeze(strf)
		r2 = np.squeeze(r2)

		if return_score:
			return strf, intercept, r2
		else:
			return strf, intercept

	def calculate_strf_for_neurons(self, 
		method, recording_idx, window_length=0.5, 
		return_scores=False, cells=None, **kwargs
	):
		'''Calculates STRF for specified neurons.

		Parameters
		----------
		method : string
			regression method to use when calculating STRFs

		recording_idx : int
			recording index to obtain design and response matrices

		window_length : float
			number of seconds to fit in STRF window

		return_scores : bool
			flag indicating whether to return explained variance over window.

		cells : int, list, np.ndarray or None
			the set of cell indices under consideration. if the first two, converts
			to a np.ndarray. if None, creates a np array of all cells.

		Returns
		-------
		strf : np.ndarray
			n_cells x n_frames_per_window x n_features array describing the
			spatio-temporal receptive field.

		intercepts : np.ndarray
			n_cells x n_frames_per_window array containing the intercepts for
			the STRFs.

		r2s : np.ndarray
			returned only if requested; n_cells x n_frames_per_window array
			containing the explained variance of the STRFfor each frame 
			in the window.
		'''
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
		elif method == 'UoILasso':
			fitter = UoI_Lasso(
				normalize=kwargs.get('normalize', True),
				estimation_score=kwargs.get('estimation_score', 'BIC'),
				n_lambdas=kwargs.get('n_lambdas', 30),
				n_boots_sel=kwargs.get('n_boots_sel', 30),
				n_boots_est=kwargs.get('n_boots_est', 30),
				selection_thres_min=kwargs.get('selection_thres_min', 1.0),
			)
		else:
			raise ValueError('method %g is not available.' %method)

		# extract dimensions and create storage
		n_features, n_samples = stimuli.shape 
		n_cells = cells.size
		strf = np.zeros((n_cells, n_frames_per_window, n_features))
		intercepts = np.zeros((n_cells, n_frames_per_window))
		# score storage
		r2s = np.zeros((n_cells, n_frames_per_window))
		aics = np.zeros((n_cells, n_frames_per_window))
		bics = np.zeros((n_cells, n_frames_per_window))

		# iterate over cells
		for cell_idx, cell in enumerate(cells):
			# copy response matrix
			responses_copy = np.copy(responses)

			# iterate over frames in window
			for frame in range(n_frames_per_window):
				print(frame)
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
				r2s[cell_idx, frame] = r2_score(
					y_true=y_true,
					y_pred=y_pred
				)

				# bics
				bics[cell_idx, frame] = self.BIC(
					y_true=y_true,
					y_pred=y_pred,
					n_features=n_features
				)

				# aics
				aics[cell_idx, frame] = self.AIC(
					y_true=y_true,
					y_pred=y_pred,
					n_features=n_features
				)

				# roll the window up
				responses_copy = np.roll(responses_copy, -1, axis=0)

		# get rid of potential unnecessary dimensions
		strf = np.squeeze(strf)
		r2s = np.squeeze(r2s)
		bics = np.squeeze(bics)
		aics = np.squeeze(aics)

		if return_scores:
			return strf, intercepts, r2s, bics, aics
		else:
			return strf, intercepts

	@staticmethod
	def BIC(y_true, y_pred, n_features):
		'''Calculate the Bayesian Information Criterion under the assumption of 
		normally distributed disturbances (which allows the BIC to take on the
		simple form below).
		
		Parameters
		----------
		Returns
		-------
		BIC : float
			Bayesian Information Criterion
		'''
		n_samples = y_true.size
		rss = np.sum((y_true - y_pred)**2)
		BIC = n_samples * np.log(rss/n_samples) + n_features * np.log(n_samples)
		return BIC

	@staticmethod
	def AIC(y_true, y_pred, n_features):
		n_samples = y_true.size
		rss = np.sum((y_true - y_pred)**2)
		AIC = n_samples * np.log(rss/n_samples) + n_features * 2
		return AIC

	@staticmethod
	def AICc(y_true, y_pred, n_features):
		n_samples = y_true.size
		rss = np.sum((y_true - y_pred)**2)
		AICc = n_samples * np.log(rss/n_samples) + n_features * 2 \
			+ 2 * (n_features**2 + n_features)/(n_samples - n_features - 1)
		return AICc