import numpy as np

from scipy.ndimage import gaussian_filter1d

from neuropacks.utils.signal import box_cox


def create_bins(t_start, t_end, bin_width, bin_type='time'):
    T = t_end - t_start
    if bin_type == 'time':
        bins = np.linspace(t_start, t_end, int(T // bin_width))
    else:
        raise ValueError('unknown bin_type')
    return bins


def bin_spike_times(spike_times_by_units, bins,
                    apply_transform={}, apply_filter={}):
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

    apply_transform: function or dictionary
        Specifies an element-wise transformation (same operation for each time)
        for the binned spike counts from each unit.

    apply_filter: function or dictionary
        Specifies a filtering operation (across the time axis)
        for the binned spike counts from each unit.

    Returns
    -------
    spike_rates: ndarray, shape (n_bins, n_units)
        Binned, transformed and filtered spike counts from each unit.
    """
    # if a transform function is provided, use it directly; otherwise build
    if not callable(apply_transform):
        if apply_transform is None:
            apply_transform = {}
        if not isinstance(apply_transform, dict):
            raise TypeError('apply_transform should be either a function or a dictionary.')
        apply_transform = build_transform(**apply_transform)

    # if a filtering function is provided, use it directly; otherwise build
    if not callable(apply_filter):
        if apply_filter is None:
            apply_filter = {}
        if not isinstance(apply_filter, dict):
            raise TypeError('apply_filter should be either a function or a dictionary.')
        apply_filter = build_filter(**apply_filter)

    # bin spike times, then apply optional transform and filter
    all_rates_list = []
    for spike_times in spike_times_by_units:
        spike_counts = np.histogram(spike_times, bins=bins)[0]
        # apply transform to spike counts at each timepoint
        spike_counts = apply_transform(spike_counts)
        # apply filter across timepoints (e.g. gaussian smoothing)
        rates = apply_filter(spike_counts)
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
    _transform : function
        Function that takes a 1D ndarray as input, and applies the same transform
        to each element of the array.
    """
    def _transform(x_traj):
        ''' x_traj is a 1D ndarray. '''
        if boxcox is not None:
            return np.array([box_cox(x, boxcox) for x in x_traj])

        if log is not None:
            return np.array([np.log(x) / np.log(log) for x in x_traj])

        # otherwise do nothing
        return x_traj

    return _transform


def build_filter(gaussian=None):
    """Build and return a function that applies filtering to a 1D timeseries.

    Parameters
    ----------
    gaussian : float (or None)
        If a value is given, apply gaussian filtering with this value as sigma.

    Returns
    -------
    _filter : function
        Function that takes a 1D ndarray as input, applies specified filtering,
        and returns the filtered array with the same shape.
    """
    def _filter(x_traj):
        ''' x_traj is a 1D ndarray. '''
        if gaussian is not None:
            return gaussian_filter1d(x_traj, sigma=gaussian)
        # otherwise do nothing
        return x_traj

    return _filter
