"""
Usage:
python extract-feature.py \
    --srcs \
        "../build/data/phm2012/initialized/Learning_set-Bearing1_1-acc.csv" \
        "../build/data/phm2012/initialized/Learning_set-Bearing1_2-acc.csv" \
    --dests \
        "../build/data/phm2012/feature-extracted/Learning_set-Bearing1_1-acc.csv" \
        "../build/data/phm2012/feature-extracted/Learning_set-Bearing1_2-acc.csv" \
    --batch-size 100
"""
import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from utils.utils import log, prepare_directory, get_args
from utils.input import get_batch

############# PAA ###############
# https://github.com/johannfaouzi/pyts

def arange(array):
    return np.arange(array[0], array[1])

def mean(ts, indices, overlapping):

    if not overlapping:
        return np.array([ts[indices[i]].mean() for i in range(indices.shape[0])])

    else:
        return np.mean(ts[indices], axis=1)

def segmentation(bounds, window_size, overlapping):

    start = bounds[:-1]
    end = bounds[1:]

    if not overlapping:
        return np.array([arange(np.array([start, end])[:, i]) for i in range(start.size)])

    else:
        correction = window_size - end + start

        new_start = start.copy()
        new_start[start.size // 2:] = start[start.size // 2:] - correction[start.size // 2:]

        new_end = end.copy()
        new_end[:end.size // 2] = end[:end.size // 2] + correction[:end.size // 2]

        return np.apply_along_axis(arange, 1, np.array([new_start, new_end]).T)

def paa(ts, ts_size, window_size, overlapping, n_segments=None, plot=False):

    if (n_segments == ts_size or window_size == 1):
        return ts

    if n_segments is None:
        quotient = ts_size // window_size
        remainder = ts_size % window_size
        n_segments = quotient if remainder == 0 else quotient + 1
    bounds = np.linspace(0, ts_size, n_segments + 1, endpoint=True).astype('int16')
    indices = segmentation(bounds, window_size, overlapping)

    if not plot:
        return mean(ts, indices, overlapping)
    else:
        return indices, mean(ts, indices, overlapping)

class PAA(BaseEstimator):
    """
    Piecewise Aggregate Approximation.
    Parameters
    ----------
    window_size : int or None (default = None)
        size of the sliding window
    output_size : int or None (default = None)
        size of the returned time series
    overlapping : bool (default = True)
        when output_size is specified, the window_size is fixed
        if overlapping is True and may vary if overlapping is False.
        Ignored if window_size is specified.
    """

    def __init__(self, window_size=None, output_size=None, overlapping=True):

        self.window_size = window_size
        self.output_size = output_size
        self.overlapping = overlapping

    def fit(self, X, y=None):
        pass

    def fit_transform(self, X, y=None):
        """Trasnform the provided data.
        Parameters
        ----------
        X : np.ndarray, shape = [n_samples, n_features]
        Returns
        -------
        X_new : np.ndarray, shape = [n_samples, new_n_features]
            Transformed data. If output_size is not None,
            new_n_features is equal to output_size. Otherwise,
            new_n_features is equal to ceil(n_features // window_size).
        """

        return self.transform(X)

    def transform(self, X):
        """Trasnform the provided data.
        Parameters
        ----------
        X : np.ndarray, shape = [n_samples, n_features]
        Returns
        -------
        X_new : np.ndarray, shape = [n_samples, n_features]
            Transformed data.
        """

        # Check input data
        if not (isinstance(X, np.ndarray) and X.ndim == 2):
            raise ValueError("'X' must be a 2-dimensional np.ndarray.")

        # Shape parameters
        n_samples, n_features = X.shape

        # Check parameters and compute window_size if output_size is given
        if (self.window_size is None and self.output_size is None):
            raise ValueError("'window_size' xor 'output_size' must be specified.")
        elif (self.window_size is not None and self.output_size is not None):
            raise ValueError("'window_size' xor 'output_size' must be specified.")
        elif (self.window_size is not None and self.output_size is None):
            if not isinstance(self.overlapping, (float, int)):
                raise TypeError("'overlapping' must be a boolean.")
            if not isinstance(self.window_size, int):
                raise TypeError("'window_size' must be an integer.")
            if self.window_size < 1:
                raise ValueError("'window_size' must be greater or equal than 1.")
            if self.window_size > n_features:
                raise ValueError("'window_size' must be lower or equal than the size of each time series.")
            window_size = self.window_size
        else:
            if not isinstance(self.overlapping, (float, int)):
                raise TypeError("'overlapping' must be a boolean.")
            if not isinstance(self.output_size, int):
                raise TypeError("'output_size' must be an integer.")
            if self.output_size < 1:
                raise ValueError("'output_size' must be greater or equal than 1.")
            if self.output_size > n_features:
                raise ValueError("'output_size' must be lower or equal than the size of each time series.")
            window_size = n_features // self.output_size
            window_size += 0 if n_features % self.output_size == 0 else 1

        return np.apply_along_axis(paa, 1, X, n_features, window_size, self.overlapping, self.output_size)
#################################

def get_fft(values, batch_size):
    yf = np.fft.fft(values)
    maximums = np.partition(-np.abs(yf[:batch_size//2]), 2)
    return -maximums[0], -maximums[1]

paa_inst = PAA(window_size=None, output_size=1, overlapping=False)
def get_paa_value(values):
    window_size = len(values)
    paa_values = paa_inst.transform(np.reshape(values, (-1, window_size)))
    return paa_values[0, 0]

def add_normalized_feature(df, feature_name):
    values = np.array(df[feature_name].values)
    minimum = np.amin(values)
    maximum = np.amax(values)
    df['normalized_' + feature_name] = (values - minimum) / (maximum - minimum)

def get_anomaly_flags(df):
    length = len(df)
    normal_length = int(length * 0.9)
    anomalous_length = length - normal_length
    normal_flags = np.repeat(0, normal_length)
    anomalous_flags = np.repeat(1, anomalous_length)
    anomaly_flags = np.concatenate((normal_flags, anomalous_flags))
    return anomaly_flags

if __name__ == '__main__':
    args = get_args()

    for src, dest in zip(args.srcs, args.dests):
        log('parsing %s ...' % (src, ))
        prepare_directory(os.path.dirname(dest))
        df_chunks = pd.read_csv(
            src,
            chunksize=args.chunk_size
        )
        df_result = pd.DataFrame({
            'avg': [],
            'max': [],
            'min': [],
            'fft1': [],
            'fft2': [],
            'paa': [],
        })
        for batch_idx, df_batch in get_batch(df_chunks, args.batch_size):
            if batch_idx % 1000 == 0:
                print(batch_idx)
            values = np.array(df_batch['x'])
            fft1, fft2 = get_fft(values, args.batch_size)
            paa_value = get_paa_value(values)
            df_result = df_result.append({
                'avg': np.mean(values),
                'max': np.amax(values),
                'min': np.amin(values),
                'fft1': fft1,
                'fft2': fft2,
                'paa': paa_value,
            }, ignore_index=True)
        add_normalized_feature(df_result, 'fft1')
        add_normalized_feature(df_result, 'paa')
        df_result['anomaly'] = get_anomaly_flags(df_result)
        df_result.to_csv(
            os.path.join(dest),
            header=True,
            index=False
        )
