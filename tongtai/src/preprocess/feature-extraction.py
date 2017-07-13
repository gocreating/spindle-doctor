# coding=UTF-8
import logging
import math
import sys
import glob
import os
import numpy as np
import pandas as pd
sys.path.insert(0,'..')
import utils
import pylab as plt

logging.basicConfig(level=logging.INFO)

SRC_DIR = '../../build/data/labeled'
DEST_DIR = '../../build/data/extracted'
CHUNK_SIZE = 1000
INPUT_CSV_COLUMNS = ['datetime', 'x', 'y', 'z', 'u', 'v', 'w', 'rul']
OUTPUT_CSV_COLUMNS = ['datetime', 'fft_peak']

# def rms_acc(df):
#     return ((
#         df['x'] ** 2 +
#         df['y'] ** 2 +
#         df['z'] ** 2
#     ) ** .5)

# def add_features(df):
#     # dfRolled = df.rolling(window=1000, min_periods=1, on='datetime')
#     d = pd.DataFrame()
#     dfSampled = df.resample('1s', on='datetime')
#     print dfSampled['datetime']
#     d['datetime'] = dfSampled['datetime']
#     d['mean_x'] = dfSampled['x'].mean()
#     d['std_x'] = dfSampled['x'].std()
#     return d

def add_features(df):
    fourier = np.fft.rfft(df['x'])
    timestep = 0.001
    # freq = abs(np.fft.rfftfreq(CHUNK_SIZE, d=timestep)) ** 2
    freqs = np.fft.rfftfreq(CHUNK_SIZE, d=timestep)
    # plt.plot(freqs, fourier)
    # plt.show()
    d = pd.DataFrame({
        'datetime': [
            df['datetime'].iloc[0],
        ],
        'fft_peak': [
            freqs[np.argmax(fourier)],
        ],
    })
    #
    # d['datetime'] =
    # d['fft_peak'] =
    # print d
    return d

def main():
    datasets = os.listdir(SRC_DIR)
    for dataset in datasets:
        filenames = glob.glob(
            os.path.join(SRC_DIR, dataset, '*.csv')
        )
        for filename in filenames:
            logging.info('parsing ' + filename + '...')
            n = utils.row_count(filename)
            dfChunks = pd.read_csv(
                filename,
                header=None,
                names=INPUT_CSV_COLUMNS,
                chunksize=CHUNK_SIZE
            )
            chunk_count = int(math.ceil(float(n) / CHUNK_SIZE))
            chunks = 0
            for dfChunk in dfChunks:
                dfChunk['datetime'] = pd.to_datetime(dfChunk['datetime'],  infer_datetime_format=True)
                chunks = chunks + 1
                logging.info('parsing chunk %d/%d' % (chunks, chunk_count))

                df = add_features(dfChunk)

                destDir = os.path.join(DEST_DIR, dataset)
                if not os.path.exists(destDir):
                    os.makedirs(destDir)
                df.to_csv(
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
