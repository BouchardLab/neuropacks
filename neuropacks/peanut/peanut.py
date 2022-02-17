import os
import numpy as np
import pandas as pd
import scipy
import pickle
from scipy.interpolate import interp1d
from scipy.ndimage import convolve1d, gaussian_filter1d


filter_dict = {'none': lambda x, **kwargs: x, 'gaussian': gaussian_filter1d}


def collect_unit_spiking_times(spike_times, meta,
                               target_region='HPc', spike_threshold=None):
    ''' Collect single units located in the target region (e.g. hippocampus)
    and optionally apply threshold on the total number of spikes.
    '''
    probes = [key for key, value in meta['nt_brain_region_dict'].items()
              if value == target_region]

    units = []
    for probe in spike_times.keys():
        probe_id = probe.split('_')[-1]
        if probe_id in probes:
            for unit, times in spike_times[probe].items():
                units.append(times)
        # else:
        #     continue

    # Apply threshold on the total number of spikes
    if spike_threshold is not None:
        spikes = [len(unit) for unit in units]
        spike_threshold_filter = [idx for idx in range(len(units))
                                  if len(units[idx]) > spike_threshold]
        units = np.array(units, dtype=object)
        units = units[spike_threshold_filter]

    return units


def create_bins(t, bin_width_ms, bin_type='time'):
    # Convert bin width to s
    bin_width = bin_width_ms / 1000

    T = t[-1] - t[0]
    if bin_type == 'time':
        bins = np.linspace(0, T, int(T // bin_width))
    return bins, bin_width


def get_spike_rates(units, bins, t0=0,
                    filter_fn='none', boxcox=0.5, **filter_kwargs):
    spike_rates = np.zeros((bins.size - 1, len(units)))

    ff = filter_dict[filter_fn]

    for i in range(len(units)):
        # translate to 0
        units[i] -= t0

        spike_counts = np.histogram(units[i], bins=bins)[0]
        if boxcox is not None:
            spike_counts = np.array([(np.power(spike_count, boxcox) - 1) / boxcox
                                     for spike_count in spike_counts])
        spike_rates_ = ff(spike_counts.astype(np.float), **filter_kwargs)

        spike_rates[:, i] = spike_rates_
    return spike_rates


def align_behavior(t, x, bins, return_bin_centers=False):
    # Offset to 0
    t -= t[0]

    bin_centers = bins + (bins[1] - bins[0])/2
    bin_centers = bin_centers[:-1]
    interpolator = interp1d(t, x)
    xaligned = interpolator(bin_centers)

    if return_bin_centers:
        return bin_centers, xaligned
    return xaligned


def load_peanut(fpath, epoch, spike_threshold, bin_width=100, bin_type='time', boxcox=0.5,
                filter_fn='none', **filter_kwargs):
    '''
    Parameters:
        fpath: str
               path to file
        epoch: int
            which epoch (session) to load.
            For day14, the rat is running during even numbered epochs.
        spike_threshold: int
              throw away neurons that spike less than the threshold during the epoch
        bin_width:     float
            Bin width for binning spikes. Note the behavior is sampled at 25ms
        bin_type : str
            Whether to bin spikes along time or position. Currently only time supported
        boxcox: float or None
            Apply boxcox transformation
        filter_fn: str
            Check filter_dict
        filter_kwargs
            keyword arguments for filter_fn
    '''

    data = pickle.load(open(fpath, 'rb'))
    dict_ = data['peanut_day14_epoch%d' % epoch]

    # Collect single units located in hippocampus,
    # and apply spike threshold
    units = collect_unit_spiking_times(
        dict_['spike_times'], dict_['identification'],
        target_region='HPc', spike_threshold=spike_threshold)

    # unpack position data
    t = dict_['position_df']['time'].values
    pos_linear = dict_['position_df']['position_linear'].values

    # create bins (the output bin_width is in seconds)
    bins, bin_width = create_bins(t, bin_width_ms=bin_width, bin_type=bin_type)

    # covnert smoothin bandwidth to indices
    if filter_fn == 'gaussian':
        filter_kwargs['sigma'] /= bin_width
        filter_kwargs['sigma'] = min(1, filter_kwargs['sigma'])

    # bin and filter to get spike rate timeseries
    spike_rates = get_spike_rates(units, bins, t0=t[0],
                                  filter_fn=filter_fn, boxcox=boxcox,
                                  **filter_kwargs)

    # Align behavior with the binned spike rates
    pos_linear = align_behavior(t, pos_linear, bins)

    dat = {}
    dat['spike_rates'] = spike_rates
    dat['pos'] = pos_linear

    return dat
