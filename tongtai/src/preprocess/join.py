# coding=UTF-8
import time
import sys
import glob
import os
import pandas as pd

SRC_DIR = '../../build/data/merged'
DEST_DIR = '../../build/data/joined'
CHANNEL_MAP = [None, 'x', 'y', 'z', 'u', 'v', 'w']
INPUT_CSV_COLUMNS = ['timestamp', 'reading']
OUTPUT_CSV_COLUMNS = ['timestamp', 'x', 'y', 'z', 'u', 'v', 'w']

def main():
    datasets = os.listdir(SRC_DIR)
    for dataset in datasets:
        workingTypes = os.listdir(os.path.join(SRC_DIR, dataset))
        for workingType in workingTypes:
            start_time = time.time()
            print('parsing ' + dataset + '/' + workingType + '...')

            readDir = os.path.join(SRC_DIR, dataset, workingType)
            dfMap = {}

            for reading in ['x', 'y', 'z', 'u', 'v', 'w']:
                dfMap[reading] = pd.read_csv(
                    os.path.join(readDir, '{0}.csv'.format(reading)),
                    names=['timestamp', reading],
                    header=None
                )

            dfAcc = dfMap['x']\
                .merge(dfMap['y'], how='inner', on='timestamp' )\
                .merge(dfMap['z'], how='inner', on='timestamp' )
            dfCurr = dfMap['u']\
                .merge(dfMap['v'], how='inner', on='timestamp' )\
                .merge(dfMap['w'], how='inner', on='timestamp' )

            maxAccIndex = len(dfAcc) - 1
            maxCurrIndex = len(dfCurr) - 1
            accIndex = 0
            currIndex = 0
            loopCount = 0
            dfOutput = pd.DataFrame(columns=OUTPUT_CSV_COLUMNS)

            while accIndex < maxAccIndex and currIndex < maxCurrIndex:
                try:
                    accEntry = dfAcc.iloc[[accIndex]]
                    currEntry = dfCurr.iloc[[currIndex]]
                    accTimestamp = accEntry.iloc[0]['timestamp']
                    currTimestamp = currEntry.iloc[0]['timestamp']

                    if loopCount % 1000 == 0:
                        print(
                            '(%d, %d) Execution time: %.3f seconds' % (accIndex, currIndex, time.time() - start_time, )
                        )
                        sys.stdout.flush()
                    loopCount += 1

                    if accTimestamp < currTimestamp:
                        accIndex += 1
                    elif accTimestamp > currTimestamp:
                        currIndex += 1
                    elif accTimestamp == currTimestamp:
                        dfOutput = dfOutput.append([{
                            'timestamp': str(accTimestamp),
                            'x': accEntry.iloc[0]['x'],
                            'y': accEntry.iloc[0]['y'],
                            'z': accEntry.iloc[0]['z'],
                            'u': currEntry.iloc[0]['u'],
                            'v': currEntry.iloc[0]['v'],
                            'w': currEntry.iloc[0]['w'],
                        }], ignore_index=True)
                        accIndex += 1
                        currIndex += 1
                except Exception as e:
                    print e
                    print(
                        '(%d, %d) Execution time: %.3f seconds' % (accIndex, currIndex, time.time() - start_time, )
                    )
                    sys.stdout.flush()

            print(
                '%s / %s - Execution time: %.3f seconds' % (dataset, workingType, time.time() - start_time, )
            )
            sys.stdout.flush()

            destDir = os.path.join(DEST_DIR, dataset)
            if not os.path.exists(destDir):
                os.makedirs(destDir)
            dfOutput.to_csv(
                os.path.join(
                    destDir,
                    '{0}.csv'.format(workingType)
                ),
                header=False,
                index=False,
                columns=OUTPUT_CSV_COLUMNS
            )

main()
