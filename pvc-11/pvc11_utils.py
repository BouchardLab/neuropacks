import h5py
import numpy as numpy
from scipy.io import loadmat

class PVC11():
	def __init__(self, data_path):
		self.data_path = data_path
		self.load_data()
		self.get_spike_counts()

	def load_data(self):
		data = loadmat(self.data_path)

		self.events = data['data'][0, 0].EVENTS
		self.n_neurons, self.n_stimuli, self.n_trials = self.events.shape

		return

	def get_spike_counts(self):
		self.spike_counts = np.zeros((self.n_neurons, self.n_stimuli, self.n_trials))

		for neuron in range(n_neurons):
			for stim in range(n_stimuli):
				for trial in range(n_trials):
					self.spike_counts[neuron, stim, trial] = \
						self.events[neuron, stim, trial].size

		return self.spike_counts

	def get_design_matrix(self, form='angle'):
		if form == 'angle':
			X = np.linspace(0, 360, self.n_stimuli + 1)[:-1]
		else:
			raise ValueError()

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