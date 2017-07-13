# coding=UTF-8
import sys
import logging
import glob
import os
import pandas as pd

SRC_DIR = '../../assets/data'
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

                prev_last_timestamp = 0
                for filename in filenames:
                    df = pd.read_csv(
                        filename,
                        names=INPUT_CSV_COLUMNS,
                        header=None
                    )
                    current_head_timestamp = df.head(1).iloc[0]['timestamp']

                    if current_head_timestamp < prev_last_timestamp:
                        print os.path.basename(filename)

                    prev_last_timestamp = df.tail(1).iloc[0]['timestamp']

main()
