# coding=UTF-8
import logging
import math
import sys
import glob
import os
import pandas as pd

logging.basicConfig(level=logging.INFO)

SRC_DIR = '../../build/data/joined'
DEST_DIR = '../../build/data/labeled'
CHUNK_SIZE = 10000
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

# http://stackoverflow.com/questions/845058/how-to-get-line-count-cheaply-in-python
def row_count(filename):
    return sum(1 for line in open(filename))

def main():
    datasets = os.listdir(SRC_DIR)
    for dataset in datasets:
        filenames = glob.glob(
            os.path.join(SRC_DIR, dataset, '*.csv')
        )
        for filename in filenames:
            logging.info('parsing ' + filename + '...')
            n = row_count(filename)
            lastRow = pd.read_csv(
                filename,
                header=None,
                skiprows=n - 1,
                names=INPUT_CSV_COLUMNS,
            )
            add_features(lastRow)
            lastTimestamp = toMillisecond(lastRow)

            dfChunks = pd.read_csv(
                filename,
                header=None,
                names=INPUT_CSV_COLUMNS,
                chunksize=CHUNK_SIZE
            )
            firstRow = dfChunks.get_chunk(1)
            add_features(firstRow)
            firstTimestamp = toMillisecond(firstRow)
            totalLife = int(lastTimestamp) - int(firstTimestamp)
            chunk_count = int(math.ceil(float(n) / CHUNK_SIZE))
            chunks = 0
            for dfChunk in dfChunks:
                chunks = chunks + 1
                logging.info('parsing chunk %d/%d' % (chunks, chunk_count))

                add_features(dfChunk)
                currentTimestamp = toMillisecond(dfChunk)
                dfChunk['rul'] = int(lastTimestamp) - currentTimestamp

                destDir = os.path.join(DEST_DIR, dataset)
                if not os.path.exists(destDir):
                    os.makedirs(destDir)
                dfChunk.to_csv(
                    os.path.join(
                        destDir,
                        os.path.basename(filename)
                    ),
                    mode='a',
                    header=False,
                    index=False,
                    columns=OUTPUT_CSV_COLUMNS
                )

main()
