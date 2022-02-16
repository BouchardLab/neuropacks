import numpy as np


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


def intersect_intervals(*intervals):
    intervals = np.concatenate([np.squeeze(np.array(i))[np.newaxis] for i in intervals], axes=0)
    interval = np.array([intervals.max(axis=0), intervals.min(axis=0)])
    if interval[0] >= interval[1]:
        raise ValueError('No intersection in given intervals.')
    return interval


def normalize_neural(data, baseline, method='zscore'):
    """Normalize neural data to baseline.

    Parameters
    ----------
    data : ndarray
        Data to normalize. Should have 1 additional axis at the 0th position compared to baseline.
    baseline : ndarray
        Baseline neural data. 0th axis should be samples to calculate normalization statistics.
    method : str
        How to normalize the neural data.
        - 'zscore' computes (data - mean) / std
        - 'ratio' computes data / mean
    """
    mean = baseline.mean(axis=0, keepdims=True)
    std = baseline.std(axis=0, keepdims=True)
    if method == 'zscore':
        data -= mean[np.newaxis]
        data /= std[np.newaxis]
    elif method == 'ratio':
        data /= mean[np.newaxis]
    else:
        raise ValueError
    return data, mean, std
