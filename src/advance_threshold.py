"""
Usage:
python advance_threshold.py \
    --srcs \
        ../build/data/phm2012/labeled/Learning_set-Bearing1_1-acc.csv \
        ../build/data/phm2012/labeled/Learning_set-Bearing1_2-acc.csv \
        ../build/data/phm2012/labeled/Learning_set-Bearing2_1-acc.csv \
        ../build/data/phm2012/labeled/Learning_set-Bearing2_2-acc.csv \
        ../build/data/phm2012/labeled/Learning_set-Bearing3_1-acc.csv \
        ../build/data/phm2012/labeled/Learning_set-Bearing3_2-acc.csv \
        ../build/data/phm2012/labeled/Full_Test_Set-Bearing1_3-acc.csv \
        ../build/data/phm2012/labeled/Full_Test_Set-Bearing1_4-acc.csv \
        ../build/data/phm2012/labeled/Full_Test_Set-Bearing1_5-acc.csv \
        ../build/data/phm2012/labeled/Full_Test_Set-Bearing1_6-acc.csv \
        ../build/data/phm2012/labeled/Full_Test_Set-Bearing1_7-acc.csv \
        ../build/data/phm2012/labeled/Full_Test_Set-Bearing2_3-acc.csv \
        ../build/data/phm2012/labeled/Full_Test_Set-Bearing2_4-acc.csv \
        ../build/data/phm2012/labeled/Full_Test_Set-Bearing2_5-acc.csv \
        ../build/data/phm2012/labeled/Full_Test_Set-Bearing2_6-acc.csv \
        ../build/data/phm2012/labeled/Full_Test_Set-Bearing2_7-acc.csv \
        ../build/data/phm2012/labeled/Full_Test_Set-Bearing3_3-acc.csv \
    --dest-dir ../build/plots/phm2012/advance_threshold \
    --batch-size 240 \
    --alarm-minutes 20 \
    --columns "mean(abs(x))" "mean(abs(y))" "mean(abs(x), abs(y))" \
    --thresholds 0.0 2.5 0.05
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.utils import get_args, log, prepare_directory
from utils.preprocess import get_datetime
from utils.input import get_batch

args = get_args()

def add_df_feature(df_chunks):
    df_featured = pd.DataFrame({
        'datetime': [],
        'mean(abs(x))': [],
        'mean(abs(y))': [],
        'mean(abs(x), abs(y))': [],
    })

    for batch_idx, df_batch in get_batch(df_chunks, args.chunk_size, args.batch_size):
        abs_x_mean = abs(df_batch['x']).mean()
        abs_y_mean = abs(df_batch['y']).mean()
        df_featured.loc[batch_idx] = [
            df_batch['datetime'].iloc[0],
            abs_x_mean,
            abs_y_mean,
            (abs_x_mean + abs_y_mean) * 0.5,
        ]
    df_featured['datetime'] = pd.to_datetime(df_featured['datetime'],  infer_datetime_format=True)

    return df_featured

def main():
    thresholds = np.arange(args.thresholds[0], args.thresholds[1], args.thresholds[2])

    table_true_alarm = np.empty([
        len(args.columns),
        len(args.srcs),
        len(thresholds),
    ])
    table_false_alarm = np.empty([
        len(args.columns),
        len(args.srcs),
        len(thresholds),
    ])

    for src_idx, src in enumerate(args.srcs):
        log('\n=====================')
        log('File %d "%s":' % (src_idx, src))

        _, last_datetime, _ = get_datetime(src)
        bound_datetime = last_datetime - pd.Timedelta(minutes=args.alarm_minutes)

        df_chunks = pd.read_csv(
            src,
            chunksize=args.chunk_size
        )
        df_featured = add_df_feature(df_chunks)

        for column_idx, column in enumerate(args.columns):
            log('\n\tFeature "%s":' % column)

            for threshold_idx, threshold in enumerate(thresholds):
                df_before_time_bound = df_featured[df_featured.datetime < bound_datetime]
                df_total_alarm = df_before_time_bound[df_before_time_bound[column] > threshold]

                n_total_alarm = len(df_total_alarm)
                n_true_alarm = 1 if n_total_alarm > 0 else 0
                n_false_alarm = n_total_alarm - n_true_alarm

                table_true_alarm[column_idx][src_idx][threshold_idx] = n_true_alarm
                table_false_alarm[column_idx][src_idx][threshold_idx] = n_false_alarm

                log(
                    '\tthreshold = %f, n_true_alarm = %d, n_total_alarm = %d' % (threshold, n_true_alarm, n_true_alarm + n_false_alarm)
                )

    log('\ntrue alarm table')
    log('================\n')
    log(table_true_alarm)
    log('\nfalse alarm table')
    log('=================\n')
    log(table_false_alarm)

    table_total_alarm = table_true_alarm + table_false_alarm
    prevent_zero_division = np.vectorize(lambda total_alarm:
        1 if total_alarm == 0 else total_alarm
    )
    table_true_alarm_indicator = table_true_alarm.astype(bool).astype(float)

    log('\ntrue alarm indicator table')
    log('==========================\n')
    log(table_true_alarm_indicator)

    for column_idx, column in enumerate(args.columns):
        true_alarms = np.average(
            table_true_alarm[column_idx],
            axis=0
        )
        total_alarms = np.average(
            table_total_alarm[column_idx],
            axis=0
        )
        true_alarm_indicators = np.average(
            table_true_alarm_indicator[column_idx],
            axis=0
        )
        log('\ntrue_alarm_indicators')
        log('=====================\n')
        log(true_alarm_indicators)

        # visualization
        dest_dir = prepare_directory(os.path.join(args.dest_dir))

        fig, ax_true_alarm_indicator = plt.subplots()
        ax_true_alarm_indicator.set_xlabel('Feature Thresholds')
        ax_true_alarm_indicator.set_ylabel('True Alarm Indicator (%)', color='blue')
        ax_true_alarm_indicator.plot(
            thresholds,
            true_alarm_indicators * 100,
            color='blue',
            marker='.'
        )
        ax_true_alarm_indicator.set_ylim([0, 100])
        ax_true_alarm_indicator.tick_params('y', colors='blue')

        ax_n_total_alarm = ax_true_alarm_indicator.twinx()
        ax_true_alarm_indicator.set_zorder(ax_n_total_alarm.get_zorder() + 1)
        ax_true_alarm_indicator.patch.set_visible(False)
        ax_n_total_alarm.yaxis.tick_right()
        ax_n_total_alarm.set_ylabel('Total Alarm Count', color='red')
        ax_n_total_alarm.tick_params('y', colors='red')
        ax_n_total_alarm.plot(
            thresholds,
            total_alarms,
            color='red',
            marker='.'
        )

        plt.title('%d minutes alarm of feature "%s"' % (args.alarm_minutes, column))
        plt.savefig(
            os.path.join(dest_dir, '%dmin-%f,%f,%fth-%s.png' % (args.alarm_minutes, args.thresholds[0], args.thresholds[1], args.thresholds[2], column)),
            dpi=400,
            format='png'
        )
        plt.clf()

main()
