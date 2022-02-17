from types import SimpleNamespace
import scipy as sp

from pynwb import NWBHDF5IO


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
            Intervals/trails tables.
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

    def _load_ecog(self):
        """Load ecog data, if available."""
        ecog = []
        with NWBHDF5IO(self.nwb_path, 'r') as io:
            nwb = io.read()
            di = nwb.processing['preprocessing'].data_interfaces
            for n in di.keys():
                if 'ecog' in n.lower():
                    ecog.append(di[n].data[:])
                    rate = di[n].rate
                    starting_time = di[n].starting_time
        if len(ecog) == 1:
            idxs = self.electrode_df['group_name'] == 'ECoG'
            good_electrodes = ~self.electrode_df['bad'].loc[idxs].values
            self.ecog = SimpleNamespace(data=ecog[0],
                                        rate=rate,
                                        good_electrodes=good_electrodes,
                                        starting_time=starting_time)
        elif len(ecog) == 0:
            pass
        else:
            raise ValueError('Multiple ECoG sources found.')

    def _load_poly(self):
        """Load polytrode data, if available."""
        poly = []
        with NWBHDF5IO(self.nwb_path, 'r') as io:
            nwb = io.read()
            di = nwb.processing['preprocessing'].data_interfaces
            for n in di.keys():
                if 'poly' in n.lower():
                    poly.append(di[n].data[:])
                    rate = di[n].rate
                    starting_time = di[n].starting_time
        if len(poly) == 1:
            idxs = self.electrode_df['group_name'] == 'Poly'
            good_electrodes = ~self.electrode_df['bad'].loc[idxs].values
            self.poly = SimpleNamespace(data=poly[0],
                                        rate=rate,
                                        good_electrodes=good_electrodes,
                                        starting_time=starting_time)
        elif len(poly) == 0:
            pass
        else:
            raise ValueError('Multiple Poly sources found.')

    def _load_stimulus_waveform(self):
        """Load the stimulus waveform."""
        with NWBHDF5IO(self.nwb_path, 'r') as io:
            nwb = io.read()
            stim = nwb.stimulus['stim_waveform']
            self.stimulus = SimpleNamespace(data=stim.data[:],
                                            rate=stim.rate,
                                            starting_time=stim.starting_time)

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
