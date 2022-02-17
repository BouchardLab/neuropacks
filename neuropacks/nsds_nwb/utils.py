import numpy as np


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
