"""
Step 1
Usage:
python merge.py
"""
import sys
sys.path.insert(0, '..')
import glob
import os
import numpy as np
import pandas as pd
from utils.utils import log, prepare_directory

SCOPE = 'tongtai'
SRC_DIR = '../../raw/tongtai'
INPUT_CSV_COLUMNS = ['timestamp', 'reading']

def main():
    datasets = os.listdir(SRC_DIR)
    for dataset in datasets:
        dataDirs = os.listdir(os.path.join(SRC_DIR, dataset))
        for dataDir in dataDirs:
            if not os.path.isdir(os.path.join(SRC_DIR, dataset, dataDir)):
                continue
            log('parsing ' + dataset + '/' + dataDir + '...')

            workingType = dataDir.split('_')[0]
            dataType = dataDir.split('_')[1]

            readDir = os.path.join(
                SRC_DIR,
                dataset,
                workingType + '_' + dataType
            )

            channel_map = [None, 'x', 'y', 'z', 'u', 'v', 'w']
            channels = [1, 2, 3] if dataType == 'acc' else [4, 5, 6]
            if dataset == '2017-07-18-168000rpm':
                channels = [4, 5, 6] if dataType == 'acc' else [1, 2, 3]
                channel_map = [None, 'u', 'v', 'w', 'x', 'y', 'z']

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
                    if dataset == '2017-07-18-168000rpm':
                        df['timestamp'] = pd.to_datetime(
                            df['timestamp'],
                            format='%m/%d/%Y %H:%M:%S.%f'
                        ).astype(np.int64) // int(1e6)
                    else:
                        df['timestamp'] = pd.to_datetime(
                            df['timestamp'],
                            format='%Y%m%d%H%M%S%f'
                        ).astype(np.int64) // int(1e6)
                    destDir = prepare_directory(os.path.join(
                        '../../build/data/', SCOPE, 'merged', dataset, workingType
                    ))
                    df.to_csv(
                        os.path.join(
                            destDir,
                            channel_map[channel] + '.csv'
                        ),
                        mode='a',
                        header=False,
                        index=False
                    )

main()
