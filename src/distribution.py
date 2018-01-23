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
    --dest "..\build\plots\phm2012\distribution\x.eps" \
    --x-label "Range of Vibration Signal" \
    --y-label "Amount" \
    --title "Vibration Distribution" \
    --thresholds -3.5 3.0 0.05 \
    --column "x"
"""
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm
from utils.utils import log, get_args, prepare_directory

mpl.rcParams['agg.path.chunksize'] = 10000

args = get_args()

if __name__ == '__main__':
    threshold_step = args.thresholds[2]
    thresholds = np.arange(args.thresholds[0], args.thresholds[1], threshold_step)
    counts = np.zeros(shape=[len(thresholds)])
    data = np.array([])
    for src in args.srcs:
        log('parsing ' + src)
        df_chunks = pd.read_csv(
            src,
            chunksize=args.chunk_size
        )
        for idx_chunk, df_chunk in enumerate(df_chunks):
            log(idx_chunk)
            data = np.concatenate((data, np.array(df_chunk[args.column].values)), axis=0)
            for idx, threshold in enumerate(thresholds[0:-1]):
                increment_value = len(df_chunk[
                    (df_chunk[args.column] >= threshold) &
                    (df_chunk[args.column] < threshold + threshold_step)
                ])
                counts[idx] = counts[idx] + increment_value
    print('min', np.amin(counts))
    print('max', np.amax(counts))
    print('std', np.std(counts))
    print('variance', np.var(counts))
    print('avg', np.mean(counts))

    plt.figure(figsize=(10, 6))
    plt.hist(data, 1000, histtype='step', color='green', label='observation')
    # plt.bar(
    #     thresholds,
    #     counts,
    #     color='green',
    #     # edgecolor='c',
    #     width=0.03
    # )
    xs = np.arange(-3, 3, 0.001)
    pdf = norm.pdf(xs, 0, 0.45)
    scaled_pdf = pdf * 2100000
    plt.plot(xs, scaled_pdf, label='theoratical', linestyle='--', color='grey')

    plt.xlabel(args.x_label, fontsize=20)
    plt.xlim([-4, 4])
    plt.ylabel(args.y_label, fontsize=20)
    plt.title(args.title, fontsize=20)
    plt.legend(fontsize=16)
    dest_dir = prepare_directory(os.path.dirname(args.dest))
    plt.savefig(
        args.dest,
        dpi=800,
        format='eps'
    )
    plt.clf()
