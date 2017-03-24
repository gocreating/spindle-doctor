# coding=UTF-8
import sys
import logging
import glob
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.rcParams['agg.path.chunksize'] = 10000

SRC_DIR = '../../build/data/predicted'
DEST_DIR = '../../build/plots/model-prediction'
INPUT_CSV_COLUMNS = [
    'timestamp', 'yyyy', 'MM', 'dd', 'hh', 'mm', 'ss', 'fff',
    'x', 'y', 'z',
    'uInG', 'vInG', 'wInG',
    'rul', 'predictedRul',
    'ss_acc',
    'ss_curr',
    'ss_acc_normal',
    'ss_curr_normal',
]
# DTYPE = {
#     'yyyy': int,
#     'MM': int,
#     'dd': int,
#     'hh': int,
#     'mm': int,
#     'ss': int,
#     'fff': int,
# }

def toMillisecond(rowOrDataframe):
    return (
        (
            rowOrDataframe['hh'] * 3600 +
            rowOrDataframe['mm'] * 60 +
            rowOrDataframe['ss']
        ) * 1000 +
        rowOrDataframe['fff']
    )

def plot(df, dataset, workingType, featureType):
    defaultMarker = {
        'c': 'red',
        'lw': 0,
        'alpha': 0.5,
    }

    plt.xlim(df['millisecond'].min(), df['millisecond'].min() + (df['millisecond'].max() - df['millisecond'].min()) * 1.1)
    plt.xlabel('Life (ms)')
    plt.ylabel('RUL (ms)')

    plt.scatter(
        df['millisecond'],
        df['predictedRul'],
        **dict(defaultMarker.items() + ({
            's': 1,
            'c': 'red',
        }).items())
    )
    # plt.plot(
    #     df['millisecond'],
    #     df['rul'],
    #     **dict(defaultMarker.items() + ({
    #         'c': 'green',
    #         'lw': 2,
    #         'alpha': 1,
    #     }).items())
    # )
    plt.scatter(
        df['millisecond'],
        df['rul'],
        **dict(defaultMarker.items() + ({
            's': 1,
            'c': 'green',
        }).items())
    )

    destDir = os.path.join(DEST_DIR, dataset, workingType)
    if not os.path.exists(destDir):
        os.makedirs(destDir)
    plt.title(featureType)
    plt.legend(loc='lower left')
    plt.savefig(
        os.path.join(destDir, '{0}.jpg'.format(featureType)),
        dpi=400,
        format='jpg'
    )
    plt.clf()

def main():
    datasets = os.listdir(SRC_DIR)
    for dataset in datasets:
        workingTypes = os.listdir(os.path.join(SRC_DIR, dataset))
        for workingType in workingTypes:
            filenames = glob.glob(os.path.join(
                SRC_DIR,
                dataset,
                workingType,
                '*.csv'
            ))
            for fileFullName in filenames:
                filename = os.path.basename(fileFullName)
                print('reading ' + filename + '...')
                featureType = filename.replace('.csv', '')

                df = pd.read_csv(
                    fileFullName,
                    sep=',',
                    header=None,
                    names=INPUT_CSV_COLUMNS
                )

                timestampInMs = toMillisecond(df)
                df['millisecond'] = timestampInMs - timestampInMs.min()
                plot(df, dataset, workingType, featureType)

main()
