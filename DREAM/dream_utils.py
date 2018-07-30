import numpy as np
import scipy.io as scio

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV

class DREAM:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = scio.loadmat(data_path, struct_as_record=False)
        self.subjects = self.data['Subject'].ravel()
        self.n_subjects = self.subjects.size
        self.n_trials = [subject.Trial.size for subject in self.subjects]

    def get_subject(self, subject_idx):
        if subject_idx >= self.n_subjects:
            raise ValueError('Subject index is not valid.')
        else:
            subject = self.subjects[subject_idx]
        return subject
    
    def get_trials(self, subject_idx):
        subject = self.get_subject(subject_idx)
        trials = subject.Trial.ravel()
        return trials
    
    def get_trial(self, subject_idx, trial_idx):
        trials = self.get_trials(subject_idx)
        n_trials = trials.size
        if trial_idx >= n_trials:
            raise ValueError('Trial index is not valid.')
        else:
            trial = trials[trial_idx]
        return trial
    
    def get_targets(self, subject_idx):
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
    
    def get_stim_onset_for_trial(self, subject_idx, trial_idx):
        # extract trial
        trial = self.get_trial(subject_idx, trial_idx)
        # get timestamps
        timestamps = self.get_timestamps_for_trial(subject_idx=subject_idx, trial_idx=trial_idx)
        # get target position
        target_position = trial.TargetPos[:, :2]
        _, center = self.get_targets(subject_idx=subject_idx)
        # extract indices for which target is on
        target_indices = ~np.isnan(trial.TargetPos)
        for idx, target in enumerate(target_position):
            if not np.all(np.isnan(target)):
                if not np.all(np.isclose(center, target)):
                    return idx
        return None   
        
    def get_hand_pos_for_trial(self, subject_idx, trial_idx):
        # extract trial
        trial = self.get_trial(subject_idx, trial_idx)
        # extract hand position
        hand_position = trial.HandPos[:, :2]
        hand_pos_x, hand_pos_y = np.split(hand_position, [1], axis=1)
        return hand_pos_x.ravel(), hand_pos_y.ravel()
    
    def get_target_for_trial(self, subject_idx, trial_idx):
        # extract trial
        trial = self.get_trial(subject_idx, trial_idx)
        target_position = trial.TargetPos[:, :2]
        _, center = self.get_targets(subject_idx=subject_idx)
        # extract indices for which target is on
        target_indices = ~np.isnan(trial.TargetPos)
        # grab unique target values
        targets = np.unique(trial.TargetPos[target_indices[:, 0]], axis=0)[:, :2]
        reach_target = [target for target in targets if not np.allclose(target, center)]
        if len(reach_target) == 0:
            return None
        else:
            return reach_target[0]
        
    def get_angle_for_trial(self, subject_idx, trial_idx):
        reach_target = self.get_target_for_trial(subject_idx, trial_idx)
        targets, center = self.get_targets(subject_idx)
        if reach_target is not None:
            # vector from center to target
            vec = reach_target - center
            # clip any tiny values
            vec[np.isclose(vec, 0)] = 0
            # calculate angle
            angle = np.round(np.arctan2(vec[1], vec[0]) * 180/np.pi)
            # change angle to range [0, 360)
            if angle < 0:
                angle += 360.
            return angle
        else:
            return reach_target
    
    def get_neurons_for_trial(self, subject_idx, trial_idx):
        # extract trial
        trial = self.get_trial(subject_idx, trial_idx)
        # get neurons
        neurons = trial.Neuron.ravel()
        return neurons
    
    def get_spikes_for_trial(self, subject_idx, trial_idx):
        # get neurons
        neurons = self.get_neurons_for_trial(subject_idx, trial_idx)
        spikes = {neuron_id : spike_times.Spike.ravel() for neuron_id, spike_times in enumerate(neurons)}
        return spikes
    
    def get_spikes_for_trial_and_neuron(self, subject_idx, trial_idx, neuron_idx):
        spikes = self.get_spikes_for_trial(subject_idx=subject_idx, trial_idx=trial_idx)
        spikes_for_neuron = spikes[neuron_idx]
        return spikes_for_neuron
    
    def get_timestamps_for_trial(self, subject_idx, trial_idx):
        trial = self.get_trial(subject_idx, trial_idx)
        timestamps = trial.Time.ravel()
        return timestamps