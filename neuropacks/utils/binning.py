import numpy as np

from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d


def spike_times_to_rates(unit_spiking_times, bins=None,
                         t_start=None, t_end=None,
                         bin_width=None, bin_type='time', bin_rep='left',
                         boxcox=0.5, filter_fn='none', **filter_kwargs):
    '''
    bin_width : float
        Bin width for binning spikes.
        Should be in the same unit as unit_spiking_times (in most cases seconds).
    bin_type : str
        Whether to bin spikes along time or position. Currently only time supported
    boxcox: float or None
        Apply boxcox transformation
    filter_fn: str
        Check filter_dict
    filter_kwargs
        keyword arguments for filter_fn
    '''
    if bins is None:
        bins = create_bins(t_start=t_start, t_end=t_end, bin_width=bin_width,
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
    return t_binned, spike_rates


def create_bins(t_start, t_end, bin_width, bin_type='time'):
    T = t_end - t_start
    if bin_type == 'time':
        bins = np.linspace(t_start, t_end, int(T // bin_width))
    else:
        raise ValueError('unknown bin_type')
    return bins


def get_spike_rates(unit_spiking_times, bins,
                    boxcox=0.5, log=None,
                    filter_fn='none', **filter_kwargs):
    if filter_fn == 'none':

        def do_nothing(x, **kwargs):
            ''' do nothing and just return the input argument '''
            return x

        _filter = do_nothing

    elif filter_fn == 'gaussian':
        _filter = gaussian_filter1d
        bin_width = bins[1] - bins[0]
        filter_kwargs['sigma'] /= bin_width  # convert to unit of bins
        # filter_kwargs['sigma'] = min(1, filter_kwargs['sigma']) # -- ??

    else:
        raise ValueError(f'unknown filter_fn: got {filter_fn}')

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


def downsample_by_interp(x, t, t_samp):
    interpolator = interp1d(t, x)
    x_samp = interpolator(t_samp)
    return x_samp
