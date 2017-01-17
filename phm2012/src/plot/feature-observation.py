# coding=UTF-8
import sys
import logging
import glob
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SRC_DIR = '../../build/data/merged/Learning_set'
DEST_DIR = '../../build/plots/feature-observation/Learning_set'
FILENAME_POSTFIXES = ['acc', 'temp']
FEATURE_NAMES = {
    'acc': ['h', 'm', 's', 'mus', 'hacc', 'vacc'],
    'temp': ['h', 'm', 's', 'dots', 'celsius'],
}
ACC_PLOTS = {
    'hacc_only': [{
        'featureName': 'hacc',
        'marker': {
            'c': 'blue',
        }
    }],
    'vacc_only': [{
        'featureName': 'vacc',
        'marker': {
            'c': 'red',
        },
    }],
    'hacc_plus_vacc': [{
        'featureName': 'hacc',
        'marker': {
            'c': 'blue',
        }
    }, {
        'featureName': 'vacc',
        'marker': {
            'c': 'red',
        },
    }],
}

def toSecond(rowOrDataframe):
    return (
        rowOrDataframe['h'] * 3600 +
        rowOrDataframe['m'] * 60 +
        rowOrDataframe['s']
    )

def plot(df, postfix, filename):
    defaultMarker = {
        's': 1,
        'c': 'blue',
        'facecolor': '0.5',
        'lw': 0,
        'alpha': 0.5,
    }
    if postfix == 'acc':
        df['timeInMuSecond'] = (
            toSecond(df) * 1000000 +
            df['mus']
        )
        for accPlot in ACC_PLOTS:
            plt.xlim(
                df['timeInMuSecond'].min(),
                df['timeInMuSecond'].max() + (
                    (
                        df['timeInMuSecond'].max() -
                        df['timeInMuSecond'].min()
                    ) *
                    0.1
                )
            )
            plt.ylim(-60, 60)
            plt.xlabel('Time (0.000001s)')
            plt.ylabel('Accelerometer Reading (g)')

            for feature in ACC_PLOTS[accPlot]:
                marker = dict(defaultMarker.items() + feature['marker'].items())
                plt.scatter(
                    df['timeInMuSecond'],
                    df[feature['featureName']],
                    **marker
                )

            destDir = os.path.join(DEST_DIR, accPlot)
            if not os.path.exists(destDir):
                os.makedirs(destDir)
            plt.title(filename)
            plt.legend(loc='upper left')
            plt.savefig(
                os.path.join(destDir, '{0}.jpg'.format(filename)),
                dpi=400,
                format='jpg'
            )
            plt.clf()
    elif postfix == 'temp':
        df['timeDotSecond'] = (
            toSecond(df) * 10 +
            df['dots']
        )
        plt.xlim(
            df['timeDotSecond'].min(),
            df['timeDotSecond'].max() + (
                (
                    df['timeDotSecond'].max() -
                    df['timeDotSecond'].min()
                ) *
                0.1
            )
        )
        plt.ylim(0, 200)
        plt.xlabel('Time (0.1s)')
        plt.ylabel('Temparature (celsius)')

        plt.scatter(
            df['timeDotSecond'],
            df['celsius'],
            **defaultMarker
        )

        destDir = os.path.join(DEST_DIR, 'temp')
        if not os.path.exists(destDir):
            os.makedirs(destDir)
        plt.title(filename)
        plt.legend(loc='upper left')
        plt.savefig(
            os.path.join(destDir, '{0}.jpg'.format(filename)),
            dpi=400,
            format='jpg'
        )
        plt.clf()

def main():
    for postfix in FILENAME_POSTFIXES:
        filenames = glob.glob(os.path.join(
            SRC_DIR,
            '*' + postfix + '.csv'
        ))
        for fileFullName in filenames:
            filename = os.path.basename(fileFullName)
            print('reading ' + filename + '...')

            df = pd.read_csv(
                fileFullName,
                sep=',',
                header=None,
                names=FEATURE_NAMES[postfix]
            )
            plot(df, postfix, filename)

main()
