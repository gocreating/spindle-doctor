# coding=UTF-8
import sys
import logging
import glob
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
mpl.rcParams['agg.path.chunksize'] = 10000

SRC_DIR = u'../../build/data/extracted'
DEST_DIR = u'../../build/plots/frequency-domain-feature-observation'
INPUT_CSV_COLUMNS = [
    'datetime',
    'fft_peak',
]

def plot(df, dataset, filename):
    firstDatetime = df['datetime'].iloc[0]
    plt.scatter(
        (df['datetime'] - firstDatetime).astype('timedelta64[s]'),
        df['fft_peak']
    )

    # output figure
    destDir = os.path.join(DEST_DIR, dataset, 'fft_peak')
    if not os.path.exists(destDir):
        os.makedirs(destDir)
    title = '{0}_{1}'.format(
        os.path.basename(filename),
        '+'.join(['fft_peak'])
    )
    plt.title(title)
    plt.savefig(
        os.path.join(destDir, '{0}.jpg'.format(title)),
        dpi=400,
        format='jpg'
    )
    plt.clf()

def main():
    datasets = os.listdir(SRC_DIR)
    for dataset in datasets:
        filenames = glob.glob(os.path.join(SRC_DIR, dataset, '*.csv'))
        for fileFullName in filenames:
            filename = os.path.basename(fileFullName)
            print('reading ' + filename + '...')

            df = pd.read_csv(
                fileFullName,
                sep=',',
                header=None,
                names=INPUT_CSV_COLUMNS
            )
            df['datetime'] = pd.to_datetime(df['datetime'],  infer_datetime_format=True)
            plot(df, dataset, filename)

main()
