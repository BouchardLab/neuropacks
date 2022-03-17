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
        filter = {}
    elif filter_fn == 'gaussian':
        bin_width = bins[1] - bins[0]
        sigma = filter_kwargs['sigma'] / bin_width  # convert to unit of bins
        # sigma = min(1, sigma) # -- ??
        filter = {'gaussian': sigma}
    else:
        raise ValueError(f'unknown filter_fn: got {filter_fn}')

    transform = {'boxcox': boxcox, 'log': log}

    spike_rates = bin_spike_times(unit_spiking_times, bins,
                                  transform=transform, filter=filter)
    return spike_rates


def bin_spike_times(spike_times_by_units, bins, transform={}, filter={}):
    """Bin the spike times from each single unit, and optionally apply
    transform and filtering on the binned spike counts.

    Parameters
    ----------
    spike_times_by_units : iterable (first over units, then over spikes)
        Either a list of arrays where each array stores spike times from a unit,
        or a 2D array with shape (n_units, spikes).

    bins : ndarray
        Timepoints at bin boundaries. This array should have shape (n_bin + 1, )
        where n_bin is the number of timepoints in the output spike_rates.

    transform: function or dictionary
        Specifies an element-wise transformation (same operation for each time)
        for the binned spike counts from each unit.

    filter: function or dictionary
        Specifies a filtering operation (across the time axis)
        for the binned spike counts from each unit.

    Returns
    -------
    spike_rates: ndarray, shape (n_bins, n_units)
        Binned, transformed and filtered spike counts from each unit.
    """
    # if a transform function is provided, use it directly; otherwise build
    if not callable(transform):
        if transform is None:
            transform = {}
        if not isinstance(transform, dict):
            raise TypeError('transform should be either a function or a dictionary.')
        transform = build_transform(**transform)

    # if a filtering function is provided, use it directly; otherwise build
    if not callable(filter):
        if filter is None:
            filter = {}
        if not isinstance(filter, dict):
            raise TypeError('filter should be either a function or a dictionary.')
        filter = build_filter(**filter)

    # bin spike times, then apply optional transform and filter
    all_rates_list = []
    for spike_times in spike_times_by_units:
        spike_counts = np.histogram(spike_times, bins=bins)[0]
        # apply transforms to spike counts at each timepoint
        spike_counts = transform(spike_counts)
        # apply filtering across timepoints (e.g. gaussian smoothing)
        rates = filter(spike_counts)
        all_rates_list.append(rates)

    spike_rates = np.stack(all_rates_list, axis=1)
    return spike_rates


def build_transform(boxcox=None, log=None, filter={}):
    """Build and return a function that transforms spike counts.

    Parameters
    ----------
    boxcox : float (or None)
        If a value is given, apply a boxcox transform with this parameter value.
    log : float (or None)
        If a value is given, take a log with this value as the log base.

    Returns
    -------
    transform : function
        Function that takes a 1D ndarray as input, and applies the same transform
        to each element of the array.
    """
    def transform(x_traj):
        ''' x_traj is a 1D ndarray. '''
        if boxcox is not None:
            return np.array([box_cox(x, boxcox) for x in x_traj])

        if log is not None:
            return np.array([np.log(x) / np.log(log) for x in x_traj])

        # otherwise do nothing
        return x_traj

    return transform


def build_filter(gaussian=None):
    """Build and return a function that applies filtering to a 1D timeseries.

    Parameters
    ----------
    gaussian : float (or None)
        If a value is given, apply gaussian filtering with this value as sigma.

    Returns
    -------
    transform : function
        Function that takes a 1D ndarray as input, applies specified filtering,
        and returns the filtered array with the same shape.
    """
    def filter(x_traj):
        ''' x_traj is a 1D ndarray. '''
        if gaussian is not None:
            return gaussian_filter1d(x_traj, sigma=gaussian)
        # otherwise do nothing
        return x_traj

    return filter


def box_cox(x, power_param):
    ''' one-parameter Box-Cox transformation '''
    return (np.power(x, power_param) - 1) / power_param


def downsample_by_interp(x, t, t_samp):
    interpolator = interp1d(t, x)
    x_samp = interpolator(t_samp)
    return x_samp
