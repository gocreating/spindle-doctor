"""
Step 2
Usage:
python join.py
"""
import sys
sys.path.insert(0, '..')
import glob
import os
import numpy as np
from utils.utils import log, prepare_directory

SRC_DIR = '../../build/data/tongtai/merged'
DEST_DIR = '../../build/data/tongtai/joined'

def readline(fd):
    raw_line = fd.readline().rstrip('\n')
    if raw_line:
        return raw_line.split(',')
    else:
        return None

def main():
    datasets = os.listdir(SRC_DIR)
    for dataset in datasets:
        workingTypes = os.listdir(os.path.join(SRC_DIR, dataset))
        for workingType in workingTypes:
            log('parsing ' + dataset + '/' + workingType + '...')
            destDir = prepare_directory(os.path.join(DEST_DIR, dataset))

            fdInputs = []
            fdOutput = open(os.path.join(
                destDir,
                '{0}.csv'.format(workingType)
            ), 'w')

            lines = []
            filenames = ['x', 'y', 'z', 'u', 'v', 'w']

            for i in range(0, 6):
                fdInputs.append(None)
                lines.append(None)
                fdInputs[i] = open(os.path.join(
                    SRC_DIR,
                    dataset,
                    workingType,
                    '{0}.csv'.format(filenames[i]))
                )
                lines[i] = readline(fdInputs[i])

            count = 0
            while all(lines):
                count = count + 1
                if count % 100000 == 0:
                    log(count)

                updateIndices = None
                if all(line[0] == lines[0][0] for line in lines):
                    updateIndices = list(range(0, 6))
                    fdOutput.write('{0},{1},{2},{3},{4},{5},{6}\n'.format(
                        lines[0][0],
                        lines[0][1],
                        lines[1][1],
                        lines[2][1],
                        lines[3][1],
                        lines[4][1],
                        lines[5][1],
                    ))
                else:
                    timestamps = np.array(lines)[:, 0].astype(np.int64)
                    updateIndices = np.where(timestamps == timestamps.min())[0]

                for updateIndex in updateIndices:
                    lines[updateIndex] = readline(fdInputs[updateIndex])

            for i in range(0, 6):
                fdInputs[i].close()
            fdOutput.close()

main()
