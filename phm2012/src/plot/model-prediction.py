# coding=UTF-8
import sys
import logging
import glob
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SRC_DIR = '../../build/data/predicted'
DEST_DIR = '../../build/plots/model-prediction'
FILENAME_POSTFIXES = ['acc', 'temp']
INPUT_CSV_COLUMNS = {
    'acc': ['hacc', 'vacc', 'rul', 'normalizedRul', 'predictedRul'],
    'temp': ['celsius', 'rul', 'normalizedRul', 'predictedRul'],
}

def toSecond(rowOrDataframe):
    return (
        rowOrDataframe['h'] * 3600 +
        rowOrDataframe['m'] * 60 +
        rowOrDataframe['s']
    )

def plot(df, postfix, featureType, filename):
    defaultMarker = {
        'c': 'red',
        'lw': 0,
        'alpha': 0.5,
    }

    plt.xlim(df.index.min(), df.index.min() + (df.index.max() - df.index.min()) * 1.1)
    plt.xlabel('Index Number')
    plt.ylabel('RUL (s)')

    plt.scatter(
        df.index.values,
        df['predictedRul'],
        **dict(defaultMarker.items() + ({
            's': 1,
            'c': 'red',
        }).items())
    )
    plt.plot(
        df.index.values,
        df['rul'],
        **dict(defaultMarker.items() + ({
            'c': 'green',
            'lw': 2,
            'alpha': 1,
        }).items())
    )

    destDir = os.path.join(DEST_DIR, featureType)
    if not os.path.exists(destDir):
        os.makedirs(destDir)
    plt.title(filename)
    plt.legend(loc='lower left')
    plt.savefig(
        os.path.join(destDir, '{0}.jpg'.format(filename)),
        dpi=400,
        format='jpg'
    )
    plt.clf()

def main():
    for postfix in FILENAME_POSTFIXES:
        featureTypes = os.listdir(SRC_DIR)

        for featureType in featureTypes:
            filenames = glob.glob(os.path.join(
                SRC_DIR,
                featureType,
                '*' + postfix + '.csv'
            ))
            for fileFullName in filenames:
                filename = os.path.basename(fileFullName)
                print('reading ' + filename + '...')

                df = pd.read_csv(
                    fileFullName,
                    sep=',',
                    header=None,
                    names=INPUT_CSV_COLUMNS[postfix]
                )
                plot(df, postfix, featureType, filename)

main()
