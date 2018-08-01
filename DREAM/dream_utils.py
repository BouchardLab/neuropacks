import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV

class DREAM:
	def __init__(self, data_path):
		self.data_path = data_path
		# grab data object
		self.data = scio.loadmat(data_path, struct_as_record=False)
		# grab subjects
		self.subjects = self.data['Subject'].ravel()
		self.n_subjects = self.subjects.size
		# extract number of trials
		self.n_trials = [subject.Trial.size for subject in self.subjects]
		# extract number of neurons
		self.n_neurons = [subject.Trial[0, 0].Neuron.size for subject in self.subjects]
		# grab targets
		self.targets = np.zeros((self.n_subjects, 8, 2))
		self.centers = np.zeros((self.n_subjects, 2))
		for subject_idx in range(self.n_subjects):
			target, center = self.get_targets(subject_idx=subject_idx)
			self.targets[subject_idx] = target
			self.centers[subject_idx] = center

	def get_subject(self, subject_idx):
		'''Returns the subject Matlab struct given an index.

		Parameters
		----------
		subject_idx : int
			the index of the desired subject.

		Returns
		-------
		subject : Matlab struct 
			the subject Matlab struct.
		'''
		# check if a valid subject index was provided
		if (subject_idx >= self.n_subjects) or (subject_idx < 0):
			raise ValueError('Subject index is not valid.')
		else:
			subject = self.subjects[subject_idx]
		return subject
	
	def get_trials(self, subject_idx):
		'''Returns an array of trial Matlab structs for a given subject.

		Parameters
		----------
		subject_idx : int
			the index of the desired subject.

		Returns
		-------
		trials : array of Matlab structs
			an array of Trial structs corresponding to the desired subject.
		'''
		# extract subject
		subject = self.get_subject(subject_idx)
		# extract trials
		trials = subject.Trial.ravel()
		return trials
	
	def get_trial(self, subject_idx, trial_idx):
		'''Returns a desired trial performed by a specified subject.

		Parameters
		----------
		subject_idx : int
			the index of the desired subject.

		trial_idx : int
			the index of the desired trial.
		'''
		# obtain all trials
		trials = self.get_trials(subject_idx)
		n_trials = trials.size
		# check if desired trial index is valid
		if (trial_idx >= n_trials) or (trial_idx < 0):
			raise ValueError('Trial index is not valid.')
		else:
			trial = trials[trial_idx]
		return trial
	
	def get_targets(self, subject_idx):
		'''Get the Cartesian coordinates for the targets in a trial.

		Parameters
		----------
		subject_idx : int
			the index of the desired subject.

		Returns
		-------
		targets : 8x2 numpy array
			the x, y positions of the eight non-center targets.
		
		center : 1x2 numpy array
			the x, y position of the center target.
		'''
		# get subject
		subject = self.get_subject(subject_idx)
		# extract trials
		trials = subject.Trial.ravel()
		n_trials = trials.size
		# initialize targets
		targets = np.array([[0, 0, 0]])
		# iterate over trials
		for trial_idx in range(n_trials):
			# grab trial
			trial = trials[trial_idx]
			# extract indices for which target is on
			target_indices = ~np.isnan(trial.TargetPos)
			# grab unique target values
			target = np.unique(trial.TargetPos[target_indices[:, 0]], axis=0)
			# add the new targets on
			targets = np.concatenate((targets, target), axis=0)
		# extract only unique targets
		targets = np.unique(targets[1:, :2], axis=0)
		# get center
		center = np.mean(targets, axis=0)
		# remove center target
		targets = np.array([target for target in targets if not np.allclose(target, center)])
		return targets, center
	
	def get_timestamps_for_trial(self, subject_idx, trial_idx):
		'''Return an array of timestamps for a given trial.

		Parameters
		----------
		subject_idx : int
			the index of the desired subject.

		trial_idx : int 
			the index of the desired trial.

		Returns
		-------
		timestamps : numpy array
			an array containing the time, in seconds, of each cursor recording.
		'''
		# grab specific trial
		trial = self.get_trial(subject_idx, trial_idx)
		# grab timestamps
		timestamps = trial.Time.ravel()
		return timestamps

	def get_stim_onset_for_trial(self, subject_idx, trial_idx):
		'''Get the index and timestamp for when the stimulus target turns on.

		Parameters
		----------
		subject_idx : int
			the index of the desired subject.

		trial_idx : int 
			the index of the desired trial.

		Returns
		-------
		idx: int
			the index of the stimulus onset; None if trial does not contain a non-center target

		timestamp : float
			the timestamp, in seconds, of stimulus onset

		target : 1x2 numpy array
			x,y coordinates for the reach target on the specified trial.
		'''
		# extract trial
		trial = self.get_trial(subject_idx, trial_idx)
		# grab center target
		center = self.centers[subject_idx]
		# get timestamps
		timestamps = self.get_timestamps_for_trial(subject_idx=subject_idx, trial_idx=trial_idx)
		# get position of targets at each timestamp (remove extraneous z coordinate)
		target_position = trial.TargetPos[:, :2]
		# iterate over targets
		for time_idx, target in enumerate(target_position):
			# check if there is no target on
			if not np.all(np.isnan(target)):
				# check if target is not equal to center
				if not np.all(np.isclose(center, target)):
					# if we're here, this is the first time a non-center target has turned on 
					return time_idx, timestamps[time_idx], target
		# if we reached here, the trial is not valid
		return None   

	def get_target_for_trial(self, subject_idx, trial_idx):
		'''Returns the target of the reach in a specified trial. 
		
		Parameters
		----------
		subject_idx : int
			the index of the desired subject.

		trial_idx : int 
			the index of the desired trial.

		Returns
		-------
		target : 1x2 numpy array
			x,y coordinates for the reach target on the specified trial.
			None if no reach target on trial.
		'''
		stim_onset = self.get_stim_onset_for_trial(subject_idx=subject_idx, trial_idx=trial_idx)
		if stim_onset is not None:
			target = stim_onset[2]
			return target
		else:
			return None
		
	def get_angle_for_trial(self, subject_idx, trial_idx):
		'''Returns the angle of the reach in a specified trial. 
		
		Parameters
		----------
		subject_idx : int
			the index of the desired subject.

		trial_idx : int 
			the index of the desired trial.

		Returns
		-------
		angle : float
			the reach of the angle, constrained to [0, 360) degrees
		'''
		reach_target = self.get_target_for_trial(subject_idx, trial_idx)
		targets = self.targets[subject_idx]
		center = self.centers[subject_idx]
		# only calculate angle if there's a valid reach target
		if reach_target is not None:
			# vector from center to target
			vec = reach_target - center
			# clip any tiny values
			vec[np.isclose(vec, 0)] = 0
			# calculate angle
			angle = np.round(np.arctan2(vec[1], vec[0]) * 180/np.pi)
			# constrain angle to range [0, 360)
			if angle < 0:
				angle += 360.
			return angle
		else:
			# if we're here, there's no non-center target in this trial
			return reach_target

	def get_hand_pos_for_trial(self, subject_idx, trial_idx):
		'''Returns the hand position over a trial.

		Parameters
		----------
		subject_idx : int
			the index of the desired subject.

		trial_idx : int 
			the index of the desired trial.

		Returns
		-------
		hand_pos_x : numpy array 
			the x-coordinates of the hand position

		hand_pos_y : numpy array
			the y-coordinates of the hand position
		'''
		# extract trial
		trial = self.get_trial(subject_idx, trial_idx)
		# extract hand position
		hand_position = trial.HandPos[:, :2]
		# split up the position into x and y coordinates
		hand_pos_x, hand_pos_y = np.split(hand_position, [1], axis=1)
		return hand_pos_x.ravel(), hand_pos_y.ravel()
	
	def get_neurons_for_trial(self, subject_idx, trial_idx):
		'''Returns the neuron Matlab structs for a given subject and trial.

		Parameters
		----------
		subject_idx : int
			the index of the desired subject.

		trial_idx : int 
			the index of the desired trial.

		Returns
		-------
		neurons : array of Matlab structs
			an array of Matlab structs containing information about the neurons
		'''
		# extract trial
		trial = self.get_trial(subject_idx, trial_idx)
		# get neurons
		neurons = trial.Neuron.ravel()
		return neurons
	
	def get_spikes_for_trial(self, subject_idx, trial_idx):
		'''Returns a dictionary of spike times over the neural population for a given subject and trial.

		Parameters
		----------
		subject_idx : int
			the index of the desired subject.

		trial_idx : int 
			the index of the desired trial.
		
		Returns
		-------
		spikes : dict
			dictionary with keys equal to neuron indices and values equal
			to numpy arrays containing spike times in seconds.
		'''
		# get neurons
		neurons = self.get_neurons_for_trial(subject_idx=subject_idx, trial_idx=trial_idx)
		spikes = {neuron_id : spike_times.Spike.ravel() for neuron_id, spike_times in enumerate(neurons)}
		return spikes
	
	def get_spikes_for_trial_and_neuron(self, subject_idx, trial_idx, neuron_idx):
		'''Returns the spike times for a given subject, trial, and neuron.

		Parameters
		----------
		subject_idx : int
			the index of the desired subject.

		trial_idx : int 
			the index of the desired trial.

		Returns
		-------
		spikers_for_neuron : numpy array
			an array containing the spike times for a given neuron.
		'''
		# grab all spikes
		spikes = self.get_spikes_for_trial(subject_idx=subject_idx, trial_idx=trial_idx)
		if (neuron_idx > self.n_neurons[subject_idx]) or (neuron_idx < 0):
			raise ValueError('Invalid neuron index.')
		# grab spikes for specified neuron
		spikes_for_neuron = spikes[neuron_idx]
		return spikes_for_neuron

	def plot_trial(self, subject_idx, trial_idx, ax=None):
		if ax is None:
			fig, ax = plt.subplots(1, 1, figsize=(10, 10))
			
		center = self.centers[subject_idx]
		reach_target = self.get_target_for_trial(subject_idx=subject_idx, trial_idx=trial_idx)
		hand_pos_x, hand_pos_y = self.get_hand_pos_for_trial(subject_idx=subject_idx, trial_idx=trial_idx)
		# plot center
		ax.scatter(center[0], center[1], color='k', s=100, zorder=10)
		# plot targets
		for target in self.targets[subject_idx]:
			if reach_target is not None:
				if np.allclose(target, reach_target):
					c = 'k'
				else:
					c = 'lightgray'
			else:
				c = 'lightgray'
			ax.scatter(target[0], target[1], color=c, s=100, zorder=10)
		# plot hand position
		ax.scatter(hand_pos_x, hand_pos_y, c=np.cos(np.arange(hand_pos_x.size) * (np.pi/hand_pos_x.size)), s=30)
		# set bounds
		ax.set_xlim([center[0] - 0.15, center[0] + 0.15])
		ax.set_ylim([center[1] - 0.15, center[1] + 0.15])
		# set colors
		ax.set_facecolor('white')
		for spine in ax.spines:
			ax.spines[spine].set_edgecolor('k')
		# labels
		ax.set_title(r'\textbf{Trial %s}' %trial_idx, fontsize=30)
		ax.set_xlabel(r'\textbf{X Position}', fontsize=30)
		ax.set_ylabel(r'\textbf{Y Position}', fontsize=30)
		# remove axis labels
		ax.set_xticks([])
		ax.set_xticklabels([])
		ax.set_yticks([])
		ax.set_yticklabels([])
		return ax