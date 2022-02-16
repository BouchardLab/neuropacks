import scipy as sp

from pynwb import NWBHDF5IO

from types import SimpleNamespace


class NSDSNWBAudio:
    def __init__(self, nwb_path):
        """Base class for loading NSDS Lab Auditory datasets stored in the NWB format.

        Parameters
        ----------
        nwb_path : str or pathlike
            Location of NWB file.

        Attributes
        ----------
        nwb_path : str or pathlike
            Location of NWB file.
        ecog : NameSpace
            Namespace for ECoG data, if present.
        poly : NameSpace
            Namespace for Poly data, if present.
        stimulus : NameSpace
            Namespace for the stimulus.
        stimulus_envelope : ndarray
            Envelope of the stimulus waveform.
        electrode_df : dataframe
            Electrode dataframe.
        intervals : dataframe
            Intervals/trais tables.
        """
        self.nwb_path = nwb_path
        self.ecog = None
        self.poly = None
        self.stimulus = None
        self._stimulus_envelope = None
        with NWBHDF5IO(self.nwb_path, 'r') as io:
            nwb = io.read()
            try:
                self.intervals = nwb.intervals['trials'].to_dataframe()
            except Exception:
                self.intervals = None
            try:
                self.electrode_df = nwb.electrodes.to_dataframe()
            except Exception:
                self.electrode_df = None

    def _load_ecog(self, load_data=True):
        """Load ecog data, if available.

        CAVEAT: When load_data is True, this loads the full recording into memory.
        """
        self.ecog = self.__load_processed_neural_data('ECoG', load_data=load_data)

    def _load_poly(self, load_data=True):
        """Load polytrode data, if available.

        CAVEAT: When load_data is True, this loads the full recording into memory.
        """
        self.poly = self.__load_processed_neural_data('Poly', load_data=load_data)

    def __load_processed_neural_data(self, data_source, load_data=False):
        '''
        Inputs:
        -------
        data_source (str):  either 'ECoG' or 'Poly'
        load_data (bool):   if True, load and store full data;
                            if False, use a placeholder data=None.

        Returns:
        --------
        A SimpleNamespace object.
        '''
        data_holder = []
        with NWBHDF5IO(self.nwb_path, 'r') as io:
            nwb = io.read()
            di = nwb.processing['preprocessing'].data_interfaces
            for n in di.keys():
                if data_source.lower() in n.lower():
                    if load_data:
                        data = di[n].data[:]
                    else:
                        data = None
                    data_holder.append(data)
                    rate = di[n].rate

        if len(data_holder) == 1:
            idxs = self.electrode_df['group_name'] == data_source
            good_electrodes = ~self.electrode_df['bad'].loc[idxs].values
            return SimpleNamespace(data=data_holder[0],
                                   rate=rate,
                                   good_electrodes=good_electrodes)
        elif len(data_holder) == 0:
            return None
        else:
            raise ValueError(f'Multiple {data_source} sources found.')

    def _load_stimulus_waveform(self):
        """Load the stimulus waveform."""
        with NWBHDF5IO(self.nwb_path, 'r') as io:
            nwb = io.read()
            self.stimulus = SimpleNamespace(data=nwb.stimulus['stim_waveform'].data[:],
                                            rate=nwb.stimulus['stim_waveform'].rate)

    @property
    def stimulus_envelope(self):
        "The stimulus envelope computed through the Hilbert Transform."
        if self._stimulus_envelope is None:
            fftd = sp.fft.fft(self.stimulus.data.astype('float32'))
            freq = sp.fft.fftfreq(self.stimulus.data.size)
            fftd[freq <= 0] = 0
            fftd[freq > 0] *= 2
            self._stimulus_envelope = abs(sp.fft.ifft(fftd))
        return self._stimulus_envelope

    def baseline_data(self):
        raise NotImplementedError
