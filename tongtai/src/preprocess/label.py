# coding=UTF-8
import sys
import glob
import os
import pandas as pd

SRC_DIR = '../../build/data/joined'
DEST_DIR = '../../build/data/labeled'
INPUT_CSV_COLUMNS = ['timestamp', 'x', 'y', 'z', 'u', 'v', 'w']
OUTPUT_CSV_COLUMNS = [
    'timestamp', 'yyyy', 'MM', 'dd', 'hh', 'mm', 'ss', 'fff',
    'x', 'y', 'z',
    'uInG', 'vInG', 'wInG',
    'rul',
    'ss_acc',
    'ss_curr',
    'ss_acc_normal',
    'ss_curr_normal',
]

def add_features(df):
    # format timestamp
    df[[
        'yyyy',
        'MM',
        'dd',
        'hh',
        'mm',
        'ss',
        'fff',
    ]] = df['timestamp'].apply(timestampSplitter)

    # transform current unit
    df['uInG'] = df['u'] * 0.004
    df['vInG'] = df['v'] * 0.004
    df['wInG'] = df['w'] * 0.004

    # compute sum of squared
    df['ss_acc'] = df['x'] ** 2 + df['y'] ** 2 + df['z'] ** 2
    df['ss_curr'] = df['uInG'] ** 2 + df['vInG'] ** 2 + df['wInG'] ** 2

    # compute normalization
    df['ss_acc_normal'] = (df['ss_acc'] - df['ss_acc'].mean()) / (df['ss_acc'].max() - df['ss_acc'].min())
    df['ss_curr_normal'] = (df['ss_curr'] - df['ss_curr'].mean()) / (df['ss_curr'].max() - df['ss_curr'].min())

def timestampSplitter(timestamp):
    return pd.Series([
        (timestamp / 10000000000000),
        (timestamp / 100000000000) % 100,
        (timestamp / 1000000000) % 100,
        (timestamp / 10000000) % 100,
        (timestamp / 100000) % 100,
        (timestamp / 1000) % 100,
        timestamp % 1000,
    ])

def toMillisecond(rowOrDataframe):
    return (
        (
            rowOrDataframe['hh'] * 3600 +
            rowOrDataframe['mm'] * 60 +
            rowOrDataframe['ss']
        ) * 1000 +
        rowOrDataframe['fff']
    )

def main():
    datasets = os.listdir(SRC_DIR)
    for dataset in datasets:
        filenames = glob.glob(
            os.path.join(SRC_DIR, dataset, '*.csv')
        )
        for filename in filenames:
            print('parsing ' + filename + '...')
            df = pd.read_csv(
                filename,
                header=None,
                names=INPUT_CSV_COLUMNS,
            )
            add_features(df)

            firstRow = df.head(1)
            lastRow = df.tail(1)
            firstTimestamp = toMillisecond(firstRow)
            lastTimestamp = toMillisecond(lastRow)
            totalLife = int(lastTimestamp) - int(firstTimestamp)
            currentTimestamp = toMillisecond(df)
            df['rul'] = int(lastTimestamp) - currentTimestamp

            destDir = os.path.join(DEST_DIR, dataset)
            if not os.path.exists(destDir):
                os.makedirs(destDir)
            df.to_csv(
                os.path.join(
                    destDir,
                    os.path.basename(filename)
                ),
                header=False,
                index=False,
                columns=OUTPUT_CSV_COLUMNS
            )

main()
