# coding=UTF-8
import glob
import os
import pandas as pd
from sklearn.model_selection import train_test_split

SRC_DIR = '../../build/data/labeled'
DEST_DIR = '../../build/data/sampled'
INPUT_CSV_COLUMNS = OUTPUT_CSV_COLUMNS = [
    'timestamp', 'yyyy', 'MM', 'dd', 'hh', 'mm', 'ss', 'fff',
    'x', 'y', 'z',
    'uInG', 'vInG', 'wInG',
    'rul',
]

def main():
    datasets = os.listdir(SRC_DIR)
    for dataset in datasets:
        filenames = glob.glob(
            os.path.join(SRC_DIR, dataset, '*.csv')
        )
        for filename in filenames:
            print('parsing ' + filename + '...')
            df = pd.read_csv(
                filename,
                header=None,
                names=INPUT_CSV_COLUMNS,
            )

            df_train, df_test = train_test_split(df, test_size=0.3)

            destDir = os.path.join(
                DEST_DIR,
                dataset,
                os.path.basename(filename).replace('.csv', '')
            )
            if not os.path.exists(destDir):
                os.makedirs(destDir)
            df_train.to_csv(
                os.path.join(
                    destDir,
                    'train.csv'
                ),
                header=False,
                index=False,
                columns=OUTPUT_CSV_COLUMNS
            )
            df_test.to_csv(
                os.path.join(
                    destDir,
                    'test.csv'
                ),
                header=False,
                index=False,
                columns=OUTPUT_CSV_COLUMNS
            )

main()
