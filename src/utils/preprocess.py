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
