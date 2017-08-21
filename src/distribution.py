r"""
Usage:
=======
python distribution.py \
    --chunk-size 500000 \
    --srcs \
        "..\build\data\phm2012\initialized\Learning_set-Bearing1_1-acc.csv" \
        "..\build\data\phm2012\initialized\Learning_set-Bearing1_2-acc.csv" \
        "..\build\data\phm2012\initialized\Learning_set-Bearing2_1-acc.csv" \
        "..\build\data\phm2012\initialized\Learning_set-Bearing2_2-acc.csv" \
        "..\build\data\phm2012\initialized\Learning_set-Bearing3_1-acc.csv" \
        "..\build\data\phm2012\initialized\Learning_set-Bearing3_2-acc.csv" \
    --dest "..\build\plots\phm2012\distribution\x.png" \
    --x-label "Range" \
    --y-label "Count" \
    --title "Distribution of x" \
    --thresholds -3.5 3.0 0.05 \
    --column "x"
"""
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
from utils.utils import log, get_args, prepare_directory

mpl.rcParams['agg.path.chunksize'] = 10000

args = get_args()

if __name__ == '__main__':
    threshold_step = args.thresholds[2]
    thresholds = np.arange(args.thresholds[0], args.thresholds[1], threshold_step)
    counts = np.zeros(shape=[len(thresholds)])

    for src in args.srcs:
        log('parsing ' + src)
        df_chunks = pd.read_csv(
            src,
            chunksize=args.chunk_size
        )
        for idx_chunk, df_chunk in enumerate(df_chunks):
            log(idx_chunk)
            for idx, threshold in enumerate(thresholds[0:-1]):
                increment_value = len(df_chunk[
                    (df_chunk[args.column] >= threshold) &
                    (df_chunk[args.column] < threshold + threshold_step)
                ])
                counts[idx] = counts[idx] + increment_value

    plt.bar(
        thresholds,
        counts,
        color='green',
        # edgecolor='c',
        width=0.03
    )
    plt.xlabel(args.x_label)
    plt.ylabel(args.y_label)
    plt.title(args.title)
    dest_dir = prepare_directory(os.path.dirname(args.dest))
    plt.savefig(
        args.dest,
        dpi=400,
        format='png'
    )
    plt.clf()
