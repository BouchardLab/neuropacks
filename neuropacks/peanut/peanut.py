import numpy as np
import scipy
import pickle
from scipy.interpolate import interp1d
from scipy.ndimage import convolve1d, gaussian_filter1d

filter_dict = {'none': lambda x, **kwargs: x, 'gaussian': gaussian_filter1d}


def align_behavior(t, x, bins):
    # Offset to 0
    t -= t[0]

    bin_centers = bins + (bins[1] - bins[0])/2
    bin_centers = bin_centers[:-1]
    interpolator = interp1d(t, x)
    xaligned = interpolator(bin_centers)

    return xaligned


def load_peanut(fpath, epoch, spike_threshold, bin_width=100, bin_type='time', boxcox=0.5,
                filter_fn='none', **filter_kwargs):
    '''
    Parameters:
        fpath: str
               path to file
        epoch: int
            which epoch (session) to load. The rat is sleeping during even numbered epochs
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

    # Collect single units located in hippocampus

    Hpc_probes = [key for key, value in dict_['identification']['nt_brain_region_dict'].items()
                  if value == 'HPc']
    units = []
    for probe in dict_['spike_times'].keys():
        probe_id = probe.split('_')[-1]
        if probe_id in Hpc_probes:
            for unit, times in dict_['spike_times'][probe].items():
                units.append(times)
        else:
            continue

    # Apply spike threshold

    spikes = [len(unit) for unit in units]
    spike_threshold_filter = [idx for idx in range(len(units))
                              if len(units[idx]) > spike_threshold]
    units = np.array(units, dtype=object)
    units = units[spike_threshold_filter]

    t = dict_['position_df']['time'].values
    T = t[-1] - t[0]
    # Convert bin width to s
    bin_width = bin_width/1000

    # covnert smoothin bandwidth to indices
    if filter_fn == 'gaussian':
        filter_kwargs['sigma'] /= bin_width
        filter_kwargs['sigma'] = min(1, filter_kwargs['sigma'])

    if bin_type == 'time':
        bins = np.linspace(0, T, int(T//bin_width))

    spike_rates = np.zeros((bins.size - 1, len(units)))

    for i in range(len(units)):
        # translate to 0
        units[i] -= t[0]

        spike_counts = np.histogram(units[i], bins=bins)[0]
        if boxcox is not None:
            spike_counts = np.array([(np.power(spike_count, boxcox) - 1)/boxcox
                                     for spike_count in spike_counts])
        spike_rates_ = filter_dict[filter_fn](spike_counts.astype(np.float), **filter_kwargs)

        spike_rates[:, i] = spike_rates_

    # Align behavior with the binned spike rates
    pos_linear = dict_['position_df']['position_linear'].values
    pos_linear = align_behavior(t, pos_linear, bins)

    dat = {}
    dat['spike_rates'] = spike_rates
    dat['pos'] = pos_linear

    return dat
