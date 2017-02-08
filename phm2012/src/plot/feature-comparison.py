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
    'hacc_plus_vacc': [{
        'featureName': 'hacc',
        'marker': {
            'color': 'blue',
        }
    }, {
        'featureName': 'vacc',
        'marker': {
            'color': 'red',
        },
    }],
    'hacc_only': [{
        'featureName': 'hacc',
        'marker': {
            'color': 'blue',
        }
    }],
    'vacc_only': [{
        'featureName': 'vacc',
        'marker': {
            'color': 'red',
        },
    }],
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
            lines = ()
            labels = ()
            defaultMarker = {
                'linewidth': 0,
                'marker': '.',
                'markersize': 1,
                'color': 'blue',
                'alpha': 0.5,
                'zorder': 10,
            }
            axisOffset = 1
            fig, axTemparature = plt.subplots()

            marker = dict(defaultMarker.items() + ({
                'linewidth': 1,
                'marker': '_',
                'color': 'black',
                'zorder': 20,
            }).items())
            lineTemparature, = axTemparature.plot(
                dfTemparature['timeDotSecond'],
                dfTemparature['celsius'],
                **marker
            )
            lines += (lineTemparature,)
            labels += ('celsius',)
            axTemparature.set_ylabel('Temparature (celsius)', color=marker['color'])
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
            axTemparature.tick_params('y', colors=marker['color'])
            axTemparature.set_xlabel('Time (0.1s)')

            for feature in ACC_PLOTS[accPlot]:
                axFeature = axTemparature.twinx()
                axTemparature.set_zorder(axFeature.get_zorder() + 1)
                axTemparature.patch.set_visible(False)
                marker = dict(defaultMarker.items() + feature['marker'].items())

                new_fixed_axis = axFeature.spines['right'].set_position(('axes', axisOffset))
                axisOffset += 0.15
                axFeature.yaxis.tick_right()
                axFeature.set_ylabel('Accelerometer Reading (g)', color=marker['color'])
                axFeature.set_xlim(
                    dfAcc['timeDotSecond'].min(),
                    dfAcc['timeDotSecond'].max() + (
                        (
                            dfAcc['timeDotSecond'].max() -
                            dfAcc['timeDotSecond'].min()
                        ) *
                        0.1
                    )
                )
                axFeature.set_ylim(-60, 60)
                axFeature.tick_params('y', colors=marker['color'])

                lineFeature, = axFeature.plot(
                    dfAcc['timeDotSecond'],
                    dfAcc[feature['featureName']],
                    **marker
                )
                lines += (lineFeature,)
                labels += (feature['featureName'],)

            fig.tight_layout()

            # output figure
            destDir = os.path.join(DEST_DIR, accPlot)
            if not os.path.exists(destDir):
                os.makedirs(destDir)
            filename = 'bearing_{0}_comparison'.format(fileId)
            plt.title(filename)
            plt.legend(lines, labels, loc='upper left')
            plt.savefig(
                os.path.join(destDir, '{0}.jpg'.format(filename)),
                dpi=400,
                format='jpg'
            )
            plt.clf()

main()
