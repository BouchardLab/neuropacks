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

    def _load_ecog(self, load_data=True):
        """Load ecog data, if available.
        """
        self.ecog = self.__get_processed_neural_data('ECoG', load_data=load_data)

    def _load_poly(self, load_data=True, start_time=None, stop_time=None):
        """Load polytrode data, if available.
        """
        self.poly = self.__get_processed_neural_data('Poly', load_data=load_data)

    def get_ecog_interval(self, start_time=None, stop_time=None):
        return self.__get_processed_neural_data('ECoG', load_data=True,
                                                 start_time=start_time,
                                                 stop_time=stop_time)

    def get_poly_interval(self, start_time=None, stop_time=None):
        return self.__get_processed_neural_data('Poly', load_data=True,
                                                 start_time=start_time,
                                                 stop_time=stop_time)

    def __get_processed_neural_data(self, data_source,
                                    load_data=True, start_time=None, stop_time=None):
        '''Load and return processed ECoG or Polytrode data, if available.

        CAVEAT:
        With default settings (load_data=True, start_time=None, stop_time=None),
        this loads the full recording into memory.

        Parameters
        ----------
        data_source : str
            Either 'ECoG' or 'Poly'.
        load_data : bool
            If True, load and store full data;
            if False, use a placeholder data=None.
        start_time : float
            Start time for loading data from an interval.
        stop_time : float
            Stop time for loading data from an interval.

        Returns
        -------
        A SimpleNamespace object.
        '''
        data_holder = []
        with NWBHDF5IO(self.nwb_path, 'r') as io:
            nwb = io.read()
            di = nwb.processing['preprocessing'].data_interfaces
            for n in di.keys():
                if data_source.lower() in n.lower():
                    rate = di[n].rate
                    starting_time = di[n].starting_time
                    if load_data:
                        data_idx = slice_interval(start_time, stop_time,
                                                  rate, t_offset=starting_time)
                        data = di[n].data[data_idx]  # first axis is for the timepoints
                        n_timepoints, n_channels, n_bands = data.shape
                        if start_time is not None:
                            starting_time = start_time
                    else:
                        data = None
                        n_timepoints, n_channels, n_bands = di[n].data.shape
                    data_holder.append(data)

        name = data_source.lower()
        if start_time is not None or stop_time is not None:
            name = f'{name}_subset'

        if len(data_holder) == 1:
            idxs = self.electrode_df['group_name'] == data_source
            good_electrodes = ~self.electrode_df['bad'].loc[idxs].values
            return SimpleNamespace(name=name,
                                   data=data_holder[0],
                                   rate=rate,
                                   n_timepoints=n_timepoints,
                                   n_channels=n_channels,
                                   n_bands=n_bands,
                                   starting_time=starting_time,
                                   good_electrodes=good_electrodes)
        elif len(data_holder) == 0:
            return None
        else:
            raise ValueError(f'Multiple {data_source} sources found.')

    def _load_stimulus_waveform(self, start_time=None, stop_time=None):
        """Load the stimulus waveform.
        """
        self.stimulus = self.__get_stimulus_waveform()

    def get_stimulus_interval(self, start_time=None, stop_time=None):
        '''
        Parameters
        ----------
        start_time : float
            Start time for loading data from an interval.
        stop_time : float
            Stop time for loading data from an interval.
        '''
        return self.__get_stimulus_waveform(start_time=start_time, stop_time=stop_time)

    def __get_stimulus_waveform(self, start_time=None, stop_time=None):
        """Load and return the stimulus waveform.

        Parameters
        ----------
        start_time : float
            Start time for loading data from an interval.
        stop_time : float
            Stop time for loading data from an interval.

        Attributes
        ----------
        data : ndarray
            With shape (n_samples,)
        rate : float
            Raw audio rate (Hz)
        n_timepoints : int
            Total number of samples in the raw audio file
        starting_time : float
            Lag, in seconds, between the start of neural recording and the start
            of the stimulus audio file. If starting_time is 10., it means the
            wav file was started 10 seconds *after* the neural recording started.
            So in the reference frame of the session time (which always starts
            with the start of neural recordings), the real time of the stimulus
            should be calculated as
                time_in_session = starting_time + i_sample / rate
            where i_sample is the index of the waveform data here.
        """
        name = 'stimulus'
        if start_time is not None or stop_time is not None:
            name = f'{name}_subset'

        with NWBHDF5IO(self.nwb_path, 'r') as io:
            nwb = io.read()
            stim_waveform = nwb.stimulus['stim_waveform']

            rate = stim_waveform.rate
            starting_time = stim_waveform.starting_time
            data_idx = slice_interval(start_time, stop_time,
                                      rate, t_offset=starting_time)
            data = stim_waveform.data[data_idx]
            n_timepoints = data.shape[0]
            if start_time is not None:
                starting_time = start_time

            return SimpleNamespace(name=name,
                                   data=data,
                                   rate=rate,
                                   n_timepoints=n_timepoints,
                                   starting_time=starting_time)

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


def slice_interval(start_time, stop_time, rate, t_offset=0):
    '''
    Returns a slice object.
    '''
    if start_time is not None and start_time < t_offset:
        raise ValueError(f'start_time is before the timeseries onset, {t_offset}.')
    if stop_time is not None and stop_time < t_offset:
        raise ValueError(f'stop_time is before the timeseries onset, {t_offset}.')
    if (start_time is not None and stop_time is not None) and start_time > stop_time:
        raise ValueError('stop_time should be later than start_time.')

    if start_time is not None:
        dt_start = start_time - t_offset
        i_start = int(dt_start * rate)
    else:
        i_start = None

    if stop_time is not None:
        dt_stop = stop_time - t_offset
        i_stop = int(dt_stop * rate)
    else:
        i_stop = None

    return slice(i_start, i_stop)
