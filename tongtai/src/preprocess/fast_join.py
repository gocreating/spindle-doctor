# coding=UTF-8
import time
import sys
import glob
import os
import numpy as np
import pandas as pd

SRC_DIR = '../../build/data/merged'
DEST_DIR = '../../build/data/joined'

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
            start_time = time.time()
            print('parsing ' + dataset + '/' + workingType + '...')

            destDir = os.path.join(DEST_DIR, dataset)
            if not os.path.exists(destDir):
                os.makedirs(destDir)

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
                if count % 1000 == 0:
                    print count

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
