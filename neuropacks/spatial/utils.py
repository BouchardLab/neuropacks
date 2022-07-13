from neuropacks.utils.binning import create_bins, bin_spike_times
from neuropacks.utils.signal import downsample_by_interp


def get_binned_times_rates_pos(t, unit_spiking_times, pos_linear,
                               bin_width=200, bin_rep='left',
                               boxcox=0.5, filter_fn='none', **filter_kwargs):
    '''
    Get binned spike rates and positions.
    Useful for spatial maze experiments where neural activity encodes space.

    bin_width : float
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
    # (ad hoc) check if time variables are in ms; if so, convert to seconds
    if bin_width >= 10:
        # guessing that this is in ms; convert to s
        bin_width /= 1000
    if 'sigma' in filter_kwargs:
        if filter_kwargs['sigma'] >= 10:
            # guessing that this is in ms; convert to seconds
            filter_kwargs['sigma'] /= 1000

    bins = create_bins(t_start=t[0], t_end=t[-1], bin_width=bin_width,
                       bin_type='time')

    t_binned = get_binned_times(bins, bin_rep=bin_rep)

    # get spike rates time series from unit spike times, by binning in time
    spike_rates = get_spike_rates(unit_spiking_times, bins,
                                  filter_fn=filter_fn,
                                  boxcox=boxcox, **filter_kwargs)

    if pos_linear is not None:
        # downsample behavior to align with the binned spike rates
        pos_binned = downsample_by_interp(pos_linear, t, t_binned)
    else:
        pos_binned = None

    return t_binned, spike_rates, pos_binned


def get_binned_times(bins, bin_rep='center'):
    # assign a representative time value to each bin
    if bin_rep == 'center':
        return (bins[1:] + bins[:-1]) / 2  # midpoints
    elif bin_rep == 'left':
        return bins[:-1]    # left endpoints
    elif bin_rep == 'right':
        return bins[1:]     # right endpoints

    raise ValueError('bin_rep should be one of (center, left, right); '
                     f'got {bin_rep}.')


def get_spike_rates(unit_spiking_times, bins,
                    boxcox=0.5, log=None,
                    filter_fn='none', **filter_kwargs):
    if filter_fn == 'none':
        apply_filter = {}
    elif filter_fn == 'gaussian':
        bin_width = bins[1] - bins[0]
        sigma = filter_kwargs['sigma'] / bin_width  # convert to unit of bins
        # sigma = min(1, sigma) # -- ??
        apply_filter = {'gaussian': sigma}
    else:
        raise ValueError(f'unknown filter_fn: got {filter_fn}')

    apply_transform = {'boxcox': boxcox, 'log': log}

    spike_rates = bin_spike_times(unit_spiking_times, bins,
                                  apply_transform=apply_transform,
                                  apply_filter=apply_filter)
    return spike_rates
