"""
Step 3
Usage:
python adapt.py
"""
import sys
sys.path.insert(0, '..')
import math
import glob
import os
import numpy as np
import pandas as pd
from utils.utils import log, prepare_directory

SCOPE = 'tongtai'
SRC_DIR = '../../build/data/tongtai/joined'
DATASETS_TO_PARSE = ['2017-07-18-168000rpm']
CHUNK_SIZE = 100000
INPUT_CSV_COLUMNS = ['timestamp', 'x', 'y', 'z', 'u', 'v', 'w']
OUTPUT_CSV_COLUMNS = ['datetime', 'x', 'y', 'z', 'u', 'v', 'w']

def main():
    dest_dir = prepare_directory(os.path.join(
        '../../build/data', SCOPE, 'initialized'
    ))

    for dataset in DATASETS_TO_PARSE:
        filenames = glob.glob(os.path.join(
            SRC_DIR,
            dataset,
            '*.csv'
        ))
        for filename in filenames:
            log('parsing ' + filename + '...')
            dfChunks = pd.read_csv(
                filename,
                header=None,
                names=INPUT_CSV_COLUMNS,
                chunksize=CHUNK_SIZE
            )

            i = 0
            for dfChunk in dfChunks:
                i = i + 1
                header =  i is 1
                mode = 'a' if i > 1 else 'w'
                log('parsing chunk %d' % i)

                dfChunk['datetime'] = pd.to_datetime(dfChunk['timestamp'], unit='ms')
                dfChunk['u'] = dfChunk['u'] * 0.1
                dfChunk['v'] = dfChunk['v'] * 0.1
                dfChunk['w'] = dfChunk['w'] * 0.1
                dfChunk['x'] = dfChunk['x'] * 0.004
                dfChunk['y'] = dfChunk['y'] * 0.004
                dfChunk['z'] = dfChunk['z'] * 0.004

                dfChunk.to_csv(
                    os.path.join(dest_dir, '%s-%s' % (dataset, os.path.basename(filename))),
                    mode=mode,
                    header=header,
                    index=False,
                    columns=OUTPUT_CSV_COLUMNS
                )

main()
