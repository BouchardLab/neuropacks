import h5py
import numpy as np
from scipy.io import loadmat

class PVC11():
	def __init__(self, data_path):
		self.data_path = data_path
		self.load_data()
		self.get_spike_counts()

	def load_data(self):
		'''Loads the data from the specified path.'''
		# load data
		data = loadmat(self.data_path, struct_as_record=False)
		# extract events
		self.events = data['data'][0, 0].EVENTS
		# extract experiment parameters
		self.n_neurons, self.n_stimuli, self.n_trials = self.events.shape
		return

	def get_spike_counts(self):
		'''Get spike counts from dataset.

		Returns
		-------
		spike_counts : np.ndarray
			n_neurons x n_stim x n_trials array containing spike counts over all
			neurons and trials.
		'''

		self.spike_counts = np.zeros((self.n_neurons, self.n_stimuli, self.n_trials))

		# iterate over all trials and neurons
		for neuron in range(self.n_neurons):
			for stim in range(self.n_stimuli):
				for trial in range(self.n_trials):
					self.spike_counts[neuron, stim, trial] = \
						# compute all spike counts in the trial
						self.events[neuron, stim, trial].size

		return self.spike_counts

	def get_design_matrix(self, form='angle'):
		'''Create design matrix according to specified form.

		Parameters
		----------
		form : string
			the structure the design matrix should take on

		Returns
		-------
		X : np.ndarray
			design matrix; first dimension is equal to number of trials
			while second dimension varies according to the desired form.
		'''
		
		if form == 'angle':
			X = np.linspace(0, 360, self.n_stimuli + 1)[:-1]
		else:
			raise ValueError('incorrect design matrix form specified.')

		return X

	def get_response_matrix(self, transform=None):
		'''Calculates response matrix corresponding to ordering grouping together
		all trials for a given stimuli. Stimuli are ordered according to increasing
		angle.

		Parameters
		----------
		transform : string
			post-processing tranform to apply to spike counts (if desired)

		Returns
		-------
		Y : np.ndarray
			nd array with shape (n_responses x n_neurons)
			where n_responses = n_stimuli * n_trials
		'''

		Y = np.reshape(
			self.spike_counts, 
			(self.n_neurons, self.n_stimuli * self.n_trials)
		).T

		if transform == 'square_root':
			Y = np.sqrt(Y)
		else:
			if transform is not None:
				raise ValueError('unknown transform requested.')

		return Y

	@staticmethod
	def BIC(y_true, y_pred, n_features):
		'''Calculate the Bayesian Information Criterion under the assumption of 
		normally distributed disturbances (which allows the BIC to take on the
		simple form below).
		
		Parameters
		----------
		y_true : np.ndarray
			array of true response values

		y_pred : np.ndarray
			array of predicted response values

		n_features : int
			number of features used in the model

		Returns
		-------
		BIC : float
			Bayesian Information Criterion
		'''

		n_samples = y_true.size
		# calculate residual sum of squares
		rss = np.sum((y_true - y_pred)**2)
		BIC = n_samples * np.log(rss/n_samples)	\
				 + n_features * np.log(n_samples)
		return BIC

	@staticmethod
	def AIC(y_true, y_pred, n_features):
		'''Calculate the Akaike Information Criterion under the assumption of 
		normally distributed disturbances. Utilizes a softer penalty on the
		model parsimony than the BIC.
		
		Parameters
		----------
		y_true : np.ndarray
			array of true response values

		y_pred : np.ndarray
			array of predicted response values

		n_features : int
			number of features used in the model

		Returns
		-------
		AIC : float
			Akaike Information Criterion
		'''

		n_samples = y_true.size
		# calculate residual sum of squares
		rss = np.sum((y_true - y_pred)**2)
		AIC = n_samples * np.log(rss/n_samples) \
				+ n_features * 2 				
		return AIC

	@staticmethod
	def AICc(y_true, y_pred, n_features):
		'''Calculate the corrected Akaike Information Criterion under the 
		assumption of normally distributed disturbances. Modifies the parsimony
		penalty. Useful in cases when the number of samples is small. 
		
		Parameters
		----------
		y_true : np.ndarray
			array of true response values

		y_pred : np.ndarray
			array of predicted response values

		n_features : int
			number of features used in the model

		Returns
		-------
		AICc : float
			corrected Akaike Information Criterion
		'''

		n_samples = y_true.size
		# calculate residual sum of squares
		rss = np.sum((y_true - y_pred)**2)
		AICc = n_samples * np.log(rss/n_samples) \
				+ n_features * 2 \
				+ 2 * (n_features**2 + n_features)/(n_samples - n_features - 1)
		return AICc