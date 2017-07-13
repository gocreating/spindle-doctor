"""
Usage:
python threshold.py \
    --src ../build/data/phm2012/labeled/Learning_set-Bearing3_1-acc.csv \
    --dest-dir ../build/plots/phm2012/threshold \
    --alarm-minutes 20 \
    --columns x y \
    --abs \
    --thresholds 0.0 0.55 0.05
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.utils import get_args, log, prepare_directory
from utils.preprocess import get_datetime

def main():
    args = get_args()
    filename = args.src

    first_datetime, last_datetime, _ = get_datetime(filename)
    bound_datetime = last_datetime - pd.Timedelta(minutes=args.alarm_minutes)

    for column in args.columns:
        log('\n=====================')
        log('Feature "%s":', column)
        thresholds = np.arange(args.thresholds[0], args.thresholds[1], args.thresholds[2])
        rates = []
        for threshold in thresholds:
            df_chunks = pd.read_csv(
                filename,
                chunksize=args.chunk_size
            )
            n_total = 0
            n_above_threshold = 0
            for chunk_idx, df_chunk in enumerate(df_chunks):
                df_chunk['datetime'] = pd.to_datetime(df_chunk['datetime'],  infer_datetime_format=True)

                if args.abs:
                    df_chunk[column] = abs(df_chunk[column])

                df_before_time_bound = df_chunk[df_chunk.datetime < bound_datetime]
                df_above_threshold = df_before_time_bound[df_before_time_bound[column] > threshold]

                n_above_threshold += len(df_above_threshold)
                n_total += len(df_before_time_bound)

            rate = float(n_above_threshold) / n_total * 100
            rates.append(rate)
            log(
                'threshold %f = %f%% (%d / %d)' % (threshold, rate, n_above_threshold, n_total)
            )

        # visualization
        basename = os.path.basename(filename)
        dest_dir = prepare_directory(os.path.join(args.dest_dir, basename))
        plt.title('%s\n%d minutes alarm of feature "%s"' % (basename, args.alarm_minutes, column))
        plt.xlabel('feature thresholds')
        plt.ylabel('rates above threshold(%)')
        plt.ylim([0, 100])
        plt.plot(thresholds, rates, 'bx-')
        plt.savefig(
            os.path.join(dest_dir, '%dmin-%s.png' % (args.alarm_minutes, column)),
            dpi=400,
            format='png'
        )
        plt.clf()

main()
