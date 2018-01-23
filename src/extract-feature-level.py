"""
Usage:
python extract-feature-level.py \
    --srcs \
        "../build/data/phm2012/feature-extracted/Learning_set-Bearing1_1-acc.csv" \
        "../build/data/phm2012/feature-extracted/Learning_set-Bearing1_2-acc.csv" \
    --dests \
        "../build/data/phm2012/feature-level-extracted/Learning_set-Bearing1_1-acc.csv" \
        "../build/data/phm2012/feature-level-extracted/Learning_set-Bearing1_2-acc.csv" \
    --src-breakpoint "../build/meta/phm2012/anomaly-detection-unequal/breakpoint-128.csv"
"""
import os
import numpy as np
import pandas as pd
from bisect import bisect_left
from utils.utils import log, prepare_directory, get_args
from utils.input import get_batch

def main():
    args = get_args()
    df_breakpoints = pd.read_csv(args.src_breakpoint)
    columns = df_breakpoints.columns.values

    for src, dest in zip(args.srcs, args.dests):
        log('parsing %s ...' % (src, ))
        prepare_directory(os.path.dirname(dest))
        df_chunks = pd.read_csv(
            src,
            chunksize=args.chunk_size
        )
        for chunk_idx, df_chunk in enumerate(df_chunks):
            if chunk_idx % 1 == 0:
                print(chunk_idx)
            for column in columns:
                df_chunk['level_' + column] = [bisect_left(df_breakpoints[column], element) for element in df_chunk[column]]

            header = chunk_idx is 0
            mode = 'a' if chunk_idx > 0 else 'w'
            df_chunk.to_csv(
                os.path.join(dest),
                mode=mode,
                header=header,
                index=False
            )

main()
