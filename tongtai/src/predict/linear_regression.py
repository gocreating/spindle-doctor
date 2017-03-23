import sys
import logging
import glob
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

SRC_DIR = '../../build/data/sampled'
DEST_DIR = '../../build/data/predicted'
INPUT_CSV_COLUMNS = [
    'timestamp', 'yyyy', 'MM', 'dd', 'hh', 'mm', 'ss', 'fff',
    'x', 'y', 'z',
    'uInG', 'vInG', 'wInG',
    'rul',
]
TRAINING_AND_TESTING_FEATURES = {
    'x': ['x'],
    'x_plus_y_plus_z': ['x', 'y', 'z'],
}
OUTPUT_CSV_COLUMNS = [
    'timestamp', 'yyyy', 'MM', 'dd', 'hh', 'mm', 'ss', 'fff',
    'x', 'y', 'z',
    'uInG', 'vInG', 'wInG',
    'rul', 'predictedRul',
]

def main():
    datasets = os.listdir(SRC_DIR)
    for dataset in datasets:
        workingTypes = os.listdir(os.path.join(SRC_DIR, dataset))
        for workingType in workingTypes:
            df_train = pd.read_csv(
                os.path.join(
                    SRC_DIR,
                    dataset,
                    workingType,
                    'train.csv'
                ),
                header=None,
                names=INPUT_CSV_COLUMNS
            )
            df_test = pd.read_csv(
                os.path.join(
                    SRC_DIR,
                    dataset,
                    workingType,
                    'test.csv'
                ),
                header=None,
                names=INPUT_CSV_COLUMNS
            )

            for featureType in TRAINING_AND_TESTING_FEATURES:
                features = TRAINING_AND_TESTING_FEATURES[featureType]
                model = LinearRegression(n_jobs=-1)
                df_train_feature = df_train[features]
                model.fit(df_train_feature.values, df_train['rul'].values)

                df_test_feature = df_test[features]
                df_test['predictedRul'] = model.predict(df_test_feature.values)

                destDir = os.path.join(DEST_DIR, dataset, workingType)
                if not os.path.exists(destDir):
                    os.makedirs(destDir)
                df_test.to_csv(
                    os.path.join(
                        destDir,
                        '{0}.csv'.format(featureType)
                    ),
                    header=False,
                    index=False,
                    columns=OUTPUT_CSV_COLUMNS
                )

main()
