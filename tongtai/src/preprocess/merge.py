# coding=UTF-8
import sys
import logging
import glob
import os
import pandas as pd

SRC_DIR = '../../assets/data'
DEST_DIR = '../../build/data/merged'
CHANNEL_MAP = [None, 'x', 'y', 'z', 'u', 'v', 'w']
INPUT_CSV_COLUMNS = ['timestamp', 'reading']

def main():
    datasets = os.listdir(SRC_DIR)
    for dataset in datasets:
        dataDirs = os.listdir(os.path.join(SRC_DIR, dataset))
        for dataDir in dataDirs:
            print('parsing ' + dataset + '/' + dataDir + '...')

            workingType = dataDir.split('_')[0]
            dataType = dataDir.split('_')[1]

            readDir = os.path.join(
                SRC_DIR,
                dataset,
                workingType + '_' + dataType
            )
            channels = [1, 2, 3] if dataType == 'acc' else [4, 5, 6]

            for channel in channels:
                filenames = glob.glob(os.path.join(
                    readDir,
                    'Channel{0}_*.csv'.format(channel)
                ))

                for filename in filenames:
                    df = pd.read_csv(
                        filename,
                        names=INPUT_CSV_COLUMNS,
                        header=None
                    )

                    destDir = os.path.join(DEST_DIR, dataset, workingType)
                    if not os.path.exists(destDir):
                        os.makedirs(destDir)
                    df.to_csv(
                        os.path.join(
                            destDir,
                            CHANNEL_MAP[channel] + '.csv'
                        ),
                        mode='a',
                        header=False,
                        index=False
                    )

main()
