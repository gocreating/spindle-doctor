import glob
import os
import math
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

SRC_DIR = '../../build/data/predicted'
DEST_DIR = '../../build/data/evaluated'
FILENAME_POSTFIXES = ['acc', 'temp']
INPUT_CSV_COLUMNS = {
    'acc': ['hacc', 'vacc', 'rul', 'normalizedRul', 'predictedRul'],
    'temp': ['celsius', 'rul', 'normalizedRul', 'predictedRul'],
}
OUTPUT_CSV_COLUMNS = ['featureType', 'name', 'RMSE', 'RSquared']

def getRMSE(df):
    return math.sqrt(((df['predictedRul'] - df['rul']) ** 2).mean(axis=0))

def getRSquared(df):
    yBar = df['rul'].mean(axis=0)
    SStot = ((df['rul'] - yBar) ** 2).sum(axis=0)
    SSres = ((df['rul'] - df['predictedRul']) ** 2).sum(axis=0)
    rSquared = 1 - (SSres / SStot)
    return rSquared

def main():
    dfEvaluation = pd.DataFrame(columns=list(OUTPUT_CSV_COLUMNS))

    for postfix in FILENAME_POSTFIXES:
        featureTypes = os.listdir(SRC_DIR)

        for featureType in featureTypes:
            filenames = glob.glob(os.path.join(
                SRC_DIR,
                featureType,
                '*_{0}.csv'.format(postfix)
            ))
            for filename in filenames:
                print('evaluating ' + filename + '...')
                dfPredicted = pd.read_csv(
                    filename,
                    header=None,
                    names=INPUT_CSV_COLUMNS[postfix]
                )
                RMSE = getRMSE(dfPredicted)
                RSquared = getRSquared(dfPredicted)
                dfEvaluation.loc[len(dfEvaluation)] = [
                    featureType,
                    os.path.basename(filename).split('.')[0],
                    RMSE,
                    RSquared,
                ]

                if not os.path.exists(DEST_DIR):
                    os.makedirs(DEST_DIR)
                dfEvaluation.to_csv(
                    os.path.join(
                        DEST_DIR,
                        'evaluate-predicted.csv'
                    ),
                    header=False,
                    index=False,
                    columns=OUTPUT_CSV_COLUMNS
                )

main()
