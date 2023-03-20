import numpy as np
from scipy.io import loadmat

grating_map = {0:-1, 2:22.5, 3:45, 4:67.5, 5:90, 6:112.5, 6:135, 8:157.5}

class V1V2():

    def __init__(self, data_path):

        """Get spike times and stimuli information for V1/V2 dataset from CRCNS.

        Parameters
        ----------
        data_path : string
            The path to the V1/V2 dataset.

        Attributes
        ----------
        data_path : string
            The path to the PVC11 dataset.
        V1_spikes :nd-array shape (trials,) with each element an array of shape (time, n_neurons)
            A numpy array containing spikes at 1 ms bin resolution in V1
        V@_spikes :nd-array shape (trials,) with each element an array of shape (time, n_neurons)
            A numpy array containing spikes at 1 ms bin resolution in V1
        grating_angle: nd-array shape (trials,)
            Contains the angle of drifting grating displayed during the trial. -1 indicates blank screen    
        """

        f = loadmat(data_path, struct_as_record=False)['neuralData'][0][0]
        
        v1_spikes = np.zeros(f.spikeRasters.shape[0], dtype=object)
        v2_spikes = np.zeros(f.spikeRasters.shape[0], dtype=object)
        grating_angle = np.zeros(f.spikeRasters.shape[0])

        for i in range(v1_spikes.size):
            v1_spikes[i] = np.array(f.spikeRasters[i, 0].todense())
            v2_spikes[i] = np.array(f.spikeRasters[i, 1].todense())
            grating_angle[i] = grating_map[f.stim[i][0]]

        self.v1_spikes = v1_spikes
        self.v2_spikes = v2_spikes
        self.grating_angle = grating_angle
