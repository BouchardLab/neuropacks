import numpy as np

from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d


def get_binned_times_rates_pos(t, unit_spiking_times, pos_linear,
                               bin_width=200, bin_rep='left',
                               boxcox=0.5, filter_fn='none', **filter_kwargs):
    '''
    spike_threshold: int
        throw away neurons that spike less than the threshold during the epoch
        default value is 0 (keep all units, including those with 0 spikes)
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
    # create bins
    if bin_width >= 10:
        # guessing that this is in ms; convert to s
        bin_width /= 1000

    bins = create_bins(t_start=t[0], t_end=t[-1], bin_width=bin_width,
                       bin_type='time')

    if bin_rep == 'center':
        t_binned = bins[:-1] + bin_width / 2  # midpoints
    elif bin_rep == 'left':
        t_binned = bins[:-1]    # left endpoints
    elif bin_rep == 'right':
        t_binned = bins[1:]     # right endpoints
    else:
        raise ValueError('bin_rep should be one of center, left or right.')

    # get spike rates time series from unit spike times
    spike_rates = get_spike_rates(unit_spiking_times, bins,
                                  filter_fn=filter_fn,
                                  boxcox=boxcox, **filter_kwargs)

    if pos_linear is not None:
        # downsample behavior to align with the binned spike rates
        pos_binned = downsample_by_interp(pos_linear, t, t_binned)
    else:
        pos_binned = None

    return t_binned, spike_rates, pos_binned


def create_bins(t_start, t_end, bin_width, bin_type='time'):
    T = t_end - t_start
    if bin_type == 'time':
        bins = np.linspace(t_start, t_end, int(T // bin_width))
    elif bin_type == 'theta':
        raise NotImplementedError
    else:
        raise ValueError('unknown bin_type')
    return bins


def downsample_by_interp(x, t, t_samp):
    interpolator = interp1d(t, x)
    x_samp = interpolator(t_samp)
    return x_samp


def get_spike_rates(unit_spiking_times, bins,
                    boxcox=0.5, log=None,
                    filter_fn='none', **filter_kwargs):
    if filter_fn == 'none':
        _filter = do_nothing
    elif filter_fn == 'gaussian':
        _filter = gaussian_filter1d
        if filter_kwargs['sigma'] >= 10:
            # guessing that this is in ms; convert to seconds
            filter_kwargs['sigma'] /= 1000
        bin_width = bins[1] - bins[0]
        filter_kwargs['sigma'] /= bin_width  # convert to unit of bins
        # filter_kwargs['sigma'] = min(1, filter_kwargs['sigma'])

    all_rates_list = []
    for spike_times in unit_spiking_times:
        spike_counts = np.histogram(spike_times, bins=bins)[0]
        if boxcox is not None:
            spike_counts = np.array(
                [box_cox(spike_count, boxcox) for spike_count in spike_counts])
        if log is not None:
            spike_counts = np.array(
                [np.log(spike_count) / np.log(log) for spike_count in spike_counts])
        rates = _filter(spike_counts.astype(np.float), **filter_kwargs)
        all_rates_list.append(rates)
    spike_rates = np.stack(all_rates_list, axis=1)
    return spike_rates


def box_cox(x, power_param):
    ''' one-parameter Box-Cox transformation '''
    return (np.power(x, power_param) - 1) / power_param


def do_nothing(x, **kwargs):
    ''' do nothing and just return the input argument '''
    return x
