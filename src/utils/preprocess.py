import numpy as np
import pandas as pd
from utils.utils import get_args, get_chunk_count

def get_datetime(filename):
    args = get_args()
    chunk_count, row_count = get_chunk_count(filename, args.chunk_size)

    df = pd.read_csv(
        filename,
        nrows=1
    )
    df['datetime'] = pd.to_datetime(df['datetime'],  infer_datetime_format=True)
    first_datetime = df['datetime'][0]

    df = pd.read_csv(
        filename,
        skiprows=row_count - 1,
        names=df.columns
    )
    df['datetime'] = pd.to_datetime(df['datetime'],  infer_datetime_format=True)
    last_datetime = df['datetime'][0]

    return first_datetime, last_datetime, chunk_count

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def get_windowed_data(raw, window_size):
    return np.array([raw[k - window_size: k] for k in range(window_size, raw.shape[0] + 1)])

def breakpoints_to_centroids(breakpoints):
    centroids = np.concatenate((
        [breakpoints[0]],
        np.array([(breakpoints[i] + breakpoints[i + 1]) / 2 for i in range(0, len(breakpoints) - 1)]),
        [breakpoints[-1]]
    ))
    return centroids
