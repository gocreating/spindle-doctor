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
DEST_DIR = '../../build/plots/feature-comparison/Learning_set'
FILENAME_POSTFIXES = ['acc', 'temp']
FEATURE_NAMES = {
    'acc': ['h', 'm', 's', 'mus', 'hacc', 'vacc'],
    'temp': ['h', 'm', 's', 'dots', 'celsius'],
}
ACC_PLOTS = {
    'hacc_only': {
        'featureName': 'hacc',
        'marker': {
            'c': 'blue',
        }
    },
    'vacc_only': {
        'featureName': 'vacc',
        'marker': {
            'c': 'red',
        },
    },
}

def toSecond(rowOrDataframe):
    return (
        rowOrDataframe['h'] * 3600 +
        rowOrDataframe['m'] * 60 +
        rowOrDataframe['s']
    )

def main():
    filenames = glob.glob(os.path.join(
        SRC_DIR,
        '*' + FILENAME_POSTFIXES[1] + '.csv'
    ))
    for filename in filenames:
        temparatureFilename = os.path.basename(filename)
        dirname = os.path.dirname(filename)
        fileId = temparatureFilename[7:10]
        accFilename = 'Bearing' + fileId + '_' + FILENAME_POSTFIXES[0] + '.csv'

        print('reading ' + fileId + '...')

        dfTemparature = pd.read_csv(
            os.path.join(SRC_DIR, temparatureFilename),
            sep=',',
            header=None,
            names=FEATURE_NAMES[FILENAME_POSTFIXES[1]]
        )
        dfAcc = pd.read_csv(
            os.path.join(SRC_DIR, accFilename),
            sep=',',
            header=None,
            names=FEATURE_NAMES[FILENAME_POSTFIXES[0]]
        )

        dfTemparature['timeDotSecond'] = (
            toSecond(dfTemparature) * 10 +
            dfTemparature['dots']
        )
        dfAcc['timeDotSecond'] = (
            toSecond(dfAcc) * 10 +
            dfAcc['mus'] / 100000
        )

        for accPlot in ACC_PLOTS:
            fig, axAcc = plt.subplots()

            lineAcc, = axAcc.plot(
                dfAcc['timeDotSecond'],
                dfAcc[ACC_PLOTS[accPlot]['featureName']],
                'b.'
            )
            axAcc.set_xlabel('Time (0.1s)')
            axAcc.set_xlim(
                dfAcc['timeDotSecond'].min(),
                dfAcc['timeDotSecond'].max() + (
                    (
                        dfAcc['timeDotSecond'].max() -
                        dfAcc['timeDotSecond'].min()
                    ) *
                    0.1
                )
            )
            axAcc.set_ylabel('Accelerometer Reading (g)', color='b')
            axAcc.set_ylim(-60, 60)
            axAcc.tick_params('y', colors='b')

            axTemparature = axAcc.twinx()
            lineTemparature, = axTemparature.plot(
                dfTemparature['timeDotSecond'],
                dfTemparature['celsius'],
                'r'
            )
            axTemparature.set_ylabel('Temparature (celsius)', color='r')
            axTemparature.set_xlim(
                dfTemparature['timeDotSecond'].min(),
                dfTemparature['timeDotSecond'].max() + (
                    (
                        dfTemparature['timeDotSecond'].max() -
                        dfTemparature['timeDotSecond'].min()
                    ) *
                    0.1
                )
            )
            axTemparature.set_ylim(0, 200)
            axTemparature.tick_params('y', colors='r')

            fig.tight_layout()

            # output figure
            destDir = os.path.join(DEST_DIR, accPlot)
            if not os.path.exists(destDir):
                os.makedirs(destDir)
            filename = 'bearing_{0}_comparison'.format(fileId)
            plt.title(filename)
            plt.legend(
                (lineAcc, lineTemparature),
                (ACC_PLOTS[accPlot]['featureName'], 'celsius'),
                loc='upper left'
            )
            plt.savefig(
                os.path.join(destDir, '{0}.jpg'.format(filename)),
                dpi=400,
                format='jpg'
            )
            plt.clf()

main()
