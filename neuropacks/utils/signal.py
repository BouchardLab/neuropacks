import numpy as np

from scipy.interpolate import interp1d


def box_cox(x, power_param):
    ''' one-parameter Box-Cox transformation '''
    return (np.power(x, power_param) - 1) / power_param


def downsample_by_interp(x, t, t_samp):
    interpolator = interp1d(t, x)
    x_samp = interpolator(t_samp)
    return x_samp
