from types import SimpleNamespace
import os
import numpy as np
import scipy as sp

from pynwb import NWBHDF5IO

from neuropacks.nsds_nwb.utils import slice_interval

NEURAL_DATA_SOURCES = {'ecog': 'ECoG',
                       'poly': 'Poly'}


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
        self.block_name = os.path.basename(nwb_path).split('.nwb')[0]

        self.ecog = None
        self.poly = None
        self.stimulus = self._get_stimulus_waveform()
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

    def get_time_array(self, num_samples, pre_stim, post_stim):
        ''' Return time array for stimulus trial given number of samples 
        in a trial and pre_stim (ms) and post_stim (ms) times'''
        trials_df = self.intervals
        trials_df = trials_df[trials_df['sb'] == 's']
        start_time = pre_stim
        stim_duration = trials_df.iloc[0]['stop_time'] - \
            trials_df.iloc[0]['start_time']
        stop_time = stim_duration + post_stim
        time = np.linspace(start_time,
                           stop_time,
                           num_samples)
        return time

    def get_electrode_positions(self, data_source):
        ''' Returns relative electrode positions for the specified data source.
        data_source : str
            Either 'ecog' or 'poly'.
        '''
        elec = self.electrode_df
        data_source_cased = NEURAL_DATA_SOURCES[data_source.lower()]
        # case sensitive!
        device_idx = (elec['group_name'] == data_source_cased)
        electrode_positions = np.array([elec['rel_x'][device_idx],
                                        elec['rel_y'][device_idx],
                                        elec['rel_z'][device_idx]])
        return electrode_positions

    def _load_ecog(self, load_data=True):
        """Load ecog data, if available.
        """
        self.ecog = self._get_processed_neural_data(
            'ecog', load_data=load_data)

    def _load_poly(self, load_data=True, start_time=None, stop_time=None):
        """Load polytrode data, if available.
        """
        self.poly = self._get_processed_neural_data(
            'poly', load_data=load_data)

    def get_ecog_interval(self, start_time=None, stop_time=None):
        return self._get_processed_neural_data('ecog', load_data=True,
                                               start_time=start_time,
                                               stop_time=stop_time)

    def get_poly_interval(self, start_time=None, stop_time=None):
        return self._get_processed_neural_data('poly', load_data=True,
                                               start_time=start_time,
                                               stop_time=stop_time)

    def get_trialized_responses(self, neural_data, in_memory=True, pre_stim=0, post_stim=0, good_electrodes_flag=True):
        data_ns = self._get_processed_neural_data(
            neural_data, load_data=in_memory)
        # get number of channels by extracting a small slice of data
        num_channels = self._get_processed_neural_data(
            neural_data,
            start_time=0,
            stop_time=1).data.shape[1]
        
        if good_electrodes_flag:
            good_electrodes = data_ns.good_electrodes
        else:
            good_electrodes = [True]*num_channels

        responses_list = []
        baselines_list = []
        for ii, row in self.intervals.iterrows():

            if row['sb'] == 's':
                start_time = row['start_time'] + pre_stim
                stop_time = row['stop_time'] + post_stim
            if row['sb'] == 'b':
                start_time = row['start_time']
                stop_time = row['stop_time']

            if in_memory:
                idx = slice_interval(start_time, stop_time,
                                     rate=data_ns.rate,
                                     t_offset=data_ns.starting_time)
                data_sliced = data_ns.data[idx]
            else:
                data_sliced = self._get_processed_neural_data(
                    neural_data,
                    start_time=start_time,
                    stop_time=stop_time).data

            if row['sb'] == 's':
                responses_list.append(data_sliced[:, good_electrodes])
            if row['sb'] == 'b':
                baselines_list.append(data_sliced[:, good_electrodes])
        return responses_list, baselines_list

    def _get_processed_neural_data(self, data_source,
                                   load_data=True, start_time=None, stop_time=None):
        '''Load and return processed ECoG or Polytrode data, if available.

        CAVEAT:
        With default settings (load_data=True, start_time=None, stop_time=None),
        this loads the full recording into memory.

        Parameters
        ----------
        data_source : str
            Either 'ecog' or 'poly'.
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
                if not n.startswith('wvlt_'):
                    continue
                if data_source in n.lower():
                    bands = di[n].bands[:]
                    rate = di[n].rate
                    starting_time = di[n].starting_time
                    if load_data:
                        data_idx = slice_interval(start_time, stop_time,
                                                  rate, t_offset=starting_time)
                        # first axis is for the timepoints
                        data = di[n].data[data_idx]
                        n_timepoints, n_channels, n_bands = data.shape
                        if start_time is not None:
                            starting_time = start_time
                    else:
                        data = None
                        n_timepoints, n_channels, n_bands = di[n].data.shape
                    data_holder.append(data)

        name = data_source
        if start_time is not None or stop_time is not None:
            name = f'{name}_subset'

        data_source_cased = NEURAL_DATA_SOURCES[data_source]
        if len(data_holder) == 1:
            idxs = self.electrode_df['group_name'] == data_source_cased
            good_electrodes = ~self.electrode_df['bad'].loc[idxs].values
            return SimpleNamespace(name=name,
                                   bands=bands,
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
            raise ValueError(f'Multiple {data_source_cased} sources found.')

    def get_stimulus_interval(self, start_time=None, stop_time=None):
        '''
        Parameters
        ----------
        start_time : float
            Start time for loading data from an interval.
        stop_time : float
            Stop time for loading data from an interval.
        '''
        return self._get_stimulus_waveform(start_time=start_time, stop_time=stop_time)

    def get_trialized_stim(self, type='waveform'):
        if self.stimulus is None:
            self.stimulus = self._get_stimulus_waveform()

        if 'wav' in type:
            stim_ = self.stimulus
        elif 'env' in type:
            stim_ = SimpleNamespace(name='stimulus_envelope',
                                    data=self.stimulus_envelope,
                                    rate=self.stimulus.rate,
                                    n_timepoints=self.stimulus_envelope.shape[0],
                                    starting_time=self.stimulus.starting_time)
        else:
            raise ValueError('unknown data type')

        stim_wav_list = []
        for ii, row in self.intervals.iterrows():
            if row['sb'] != 's':
                continue

            idx = slice_interval(row['start_time'], row['stop_time'],
                                 rate=stim_.rate,
                                 t_offset=stim_.starting_time)

            stim_sliced = SimpleNamespace(name=f'stim_trial{ii}',
                                          data=stim_.data[idx],
                                          rate=stim_.rate,
                                          n_timepoints=np.sum(idx),
                                          starting_time=stim_.starting_time)
            stim_wav_list.append(stim_sliced)
        # list of namespace objects
        return stim_wav_list

    def _get_stimulus_waveform(self, start_time=None, stop_time=None):
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
