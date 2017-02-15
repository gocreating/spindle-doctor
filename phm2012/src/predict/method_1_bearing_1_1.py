import sys
import logging
import glob
import os
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

TRAINING_DIR = '../../build/data/labeled/Learning_set'
TESTING_DIR = '../../build/data/labeled/Full_Test_Set'
DEST_DIR = '../../build/data/predicted'
FILENAME_POSTFIXES = ['acc', 'temp']
TRAINING_INPUT_CSV_COLUMNS = {
    'acc': ['hacc', 'vacc', 'rul', 'normalizedRul'],
    'temp': ['celsius', 'rul', 'normalizedRul'],
}
TESTING_INPUT_CSV_COLUMNS = {
    'acc': ['hacc', 'vacc', 'rul', 'normalizedRul'],
    'temp': ['celsius', 'rul', 'normalizedRul'],
}
TRAINING_AND_TESTING_FEATURES = {
    'acc': {
        'hacc': ['hacc'],
        'vacc': ['vacc'],
        'hacc_plus_vacc': ['hacc', 'vacc'],
        'hacc_abs': ['hacc_abs'],
        'vacc_abs': ['vacc_abs'],
        'hacc_abs_plus_vacc_abs': ['hacc_abs', 'vacc_abs'],
    },
    'temp': {
        'temp': ['celsius'],
    },
}
OUTPUT_CSV_COLUMNS = {
    'acc': ['hacc', 'vacc', 'rul', 'normalizedRul', 'predictedRul'],
    'temp': ['celsius', 'rul', 'normalizedRul', 'predictedRul'],
}

def getModel(trainingDir, postfix, features):
    # model = MLPClassifier(
    #     solver='lbfgs',
    #     alpha=1e-5,
    #     hidden_layer_sizes=(5, 2),
    #     random_state=1
    # )
    model = LinearRegression(n_jobs=-1)
    dfTrain = pd.read_csv(
        os.path.join(trainingDir, 'Bearing1_1_{0}.csv'.format(postfix)),
        header=None,
        names=TRAINING_INPUT_CSV_COLUMNS[postfix]
    )
    if postfix == 'acc':
        dfTrain['hacc_abs'] = dfTrain['hacc'].abs()
        dfTrain['vacc_abs'] = dfTrain['hacc'].abs()

    dfFeature = dfTrain[features]
    model.fit(dfFeature.values, dfTrain['rul'].values)
    return model

def main():
    for postfix in FILENAME_POSTFIXES:
        filenames = glob.glob(
            os.path.join(TESTING_DIR, '*_{0}.csv'.format(postfix))
        )
        for featureType in TRAINING_AND_TESTING_FEATURES[postfix]:
            features = TRAINING_AND_TESTING_FEATURES[postfix][featureType]
            model = getModel(TRAINING_DIR, postfix, features)

            for filename in filenames:
                print('predicting ' + filename + '...')

                dfTesting = pd.read_csv(
                    filename,
                    header=None,
                    names=TESTING_INPUT_CSV_COLUMNS[postfix]
                )
                if postfix == 'acc':
                    dfTesting['hacc_abs'] = dfTesting['hacc'].abs()
                    dfTesting['vacc_abs'] = dfTesting['hacc'].abs()
                dfFeature = dfTesting[features]
                dfTesting['predictedRul'] = model.predict(dfFeature.values)

                destDir = os.path.join(DEST_DIR, featureType)
                if not os.path.exists(destDir):
                    os.makedirs(destDir)
                dfTesting.to_csv(
                    os.path.join(
                        destDir,
                        os.path.basename(filename)
                    ),
                    header=False,
                    index=False,
                    columns=OUTPUT_CSV_COLUMNS[postfix]
                )

main()
