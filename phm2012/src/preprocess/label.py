import sys
import logging
import glob
import os
import pandas as pd

SRC_DIR = '../../build/data/merged'
DEST_DIR = '../../build/data/labeled'
DATASETS = ['Learning_set', 'Full_Test_Set']
FILENAME_POSTFIXES = ['acc', 'temp']
INPUT_CSV_COLUMNS = {
    'acc': ['h', 'm', 's', 'mus', 'hacc', 'vacc'],
    'temp': ['h', 'm', 's', 'dots', 'celsius'],
}
OUTPUT_CSV_COLUMNS = {
    'acc': ['hacc', 'vacc', 'rul', 'normalizedRul'],
    'temp': ['celsius', 'rul', 'normalizedRul'],
}

def toSecond(rowOrDataframe):
    return (
        rowOrDataframe['h'] * 3600 +
        rowOrDataframe['m'] * 60 +
        rowOrDataframe['s']
    )

def main():
    for dataset in DATASETS:
        for postfix in FILENAME_POSTFIXES:
            filenames = glob.glob(
                os.path.join(SRC_DIR, dataset, '*_' + postfix +'.csv')
            )
            for filename in filenames:
                print('parsing ' + filename + '...')

                df = pd.read_csv(
                    filename,
                    header=None,
                    names=INPUT_CSV_COLUMNS[postfix]
                )
                firstRow = df.head(1)
                lastRow = df.tail(1)
                firstTimestamp = toSecond(firstRow)
                lastTimestamp = toSecond(lastRow)
                totalLife = int(lastTimestamp) - int(firstTimestamp)
                currentTimestamp = toSecond(df)
                df['rul'] = int(lastTimestamp) - currentTimestamp
                df['normalizedRul'] = df['rul'] / totalLife

                destDir = os.path.join(DEST_DIR, dataset)
                if not os.path.exists(destDir):
                    os.makedirs(destDir)
                df.to_csv(
                    os.path.join(
                        destDir,
                        os.path.basename(filename)
                    ),
                    header=False,
                    index=False,
                    columns=OUTPUT_CSV_COLUMNS[postfix]
                )

main()
