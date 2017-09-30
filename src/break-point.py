r"""
Usage:
=======
python break-point.py \
    --chunk-size 500000 \
    --srcs \
        "..\build\data\phm2012\initialized\Learning_set-Bearing1_1-acc.csv" \
        "..\build\data\phm2012\initialized\Learning_set-Bearing1_2-acc.csv" \
        "..\build\data\phm2012\initialized\Learning_set-Bearing2_1-acc.csv" \
        "..\build\data\phm2012\initialized\Learning_set-Bearing2_2-acc.csv" \
        "..\build\data\phm2012\initialized\Learning_set-Bearing3_1-acc.csv" \
        "..\build\data\phm2012\initialized\Learning_set-Bearing3_2-acc.csv" \
    --dest-dir "..\build\meta\phm2012\anomaly-detection-unequal" \
    --columns "x" "y" \
    --symbol-size 132
"""
import os
from scipy.stats.distributions import norm
import numpy as np
import pandas as pd
import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
from utils.utils import log, get_args, prepare_directory

mpl.rcParams['agg.path.chunksize'] = 10000

args = get_args()

if __name__ == '__main__':
    breakpoints = {}
    for column in args.columns:
        lens = 0
        sums = 0
        mean = 0
        sumOfSquaredDifference = 0
        std = 0
        maxValue = -float("inf")
        minValue = float("inf")

        for src in args.srcs:
            log('phase 1 of ' + src)
            # check the correctness
            # df = pd.read_csv(src)
            # print(len(df[column]))
            # print(df[column].sum())
            # print(df[column].mean())
            # print(df[column].std())
            # print(df[column].min())
            # print(df[column].max())
            df_chunks = pd.read_csv(
                src,
                chunksize=args.chunk_size
            )
            for idx_chunk, df_chunk in enumerate(df_chunks):
                log(idx_chunk)
                lens = lens + len(df_chunk)
                sums = sums + df_chunk[column].sum()
                localMin = df_chunk[column].min()
                localMax = df_chunk[column].max()
                if localMin < minValue:
                    minValue = localMin
                if localMax > maxValue:
                    maxValue = localMax

        mean = sums / lens

        for src in args.srcs:
            log('phase 2 of ' + src)
            df_chunks = pd.read_csv(
                src,
                chunksize=args.chunk_size
            )
            for idx_chunk, df_chunk in enumerate(df_chunks):
                log(idx_chunk)
                sumOfSquaredDifference = sumOfSquaredDifference + (
                    ((df_chunk[column] - mean) ** 2).sum()
                )

        std = (sumOfSquaredDifference / lens) ** 0.5

        n = norm(mean, std)
        step = (0.9999 - 0.0001) / (args.symbol_size - 2)
        breakpoints[column] = []
        for probability in np.arange(0.0001, 0.9999, step):
            # ppf(percent point function) is the inverse function of pdf(probability density function)
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html
            breakpoints[column].append(n.ppf(probability))

        print('==== Report ====')
        print('lens\t', lens)
        print('sums\t', sums)
        print('mean\t', mean)
        print('std\t', std)
        print('minValue\t', minValue)
        print('maxValue\t', maxValue)
        print('step\t', step)
        print('len(breakpoints)\t', len(breakpoints[column]))
        print('breakpoints\t', breakpoints[column])

    df_breakpoints = pd.DataFrame(breakpoints)
    dest_dir = prepare_directory(args.dest_dir)
    df_breakpoints.to_csv(
        os.path.join(
            dest_dir,
            'breakpoint.csv'
        ),
        header=True,
        index=False
    )
