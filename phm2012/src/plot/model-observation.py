import os
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

TRAINING_DIR = '../../build/data/labeled/Learning_set'
DEST_DIR = '../../build/plots/model-observation'
TRAINING_INPUT_CSV_COLUMNS = [
    'hacc', 'vacc', 'rul', 'normalizedRul'
]

def main():
    model = LinearRegression(n_jobs=-1)
    dfTrain = pd.read_csv(
        os.path.join(TRAINING_DIR, 'Bearing1_1_acc.csv'),
        header=None,
        names=TRAINING_INPUT_CSV_COLUMNS
    )
    dfTrain['hacc_abs'] = dfTrain['hacc'].abs()
    dfTrain['vacc_abs'] = dfTrain['hacc'].abs()

    model.fit(dfTrain[['hacc_abs']].values, dfTrain['rul'].values)

    # trending line
    coef = model.coef_[0]
    intercept = model.intercept_

    coordinates = [{
        'title': 'overview',
        'x1': dfTrain['hacc_abs'].min(),
        'x2': dfTrain['hacc_abs'].max(),
        'y1': dfTrain['hacc_abs'].min() * coef + intercept,
        'y2': dfTrain['hacc_abs'].max() * coef + intercept,
    }, {
        'title': 'high_resolution',
        'x1': dfTrain['hacc_abs'].min(),
        'y1': dfTrain['hacc_abs'].min() * coef + intercept,
        'y2': 0,
        'x2': (0 - intercept) / coef,
    }]

    for c in coordinates:
        lineAcc = plt.scatter(
            dfTrain['hacc_abs'].values,
            dfTrain['rul'].values,
            s=1,
            lw=0,
            alpha=0.5
        )
        lineModel, = plt.plot(
            [c['x1'], c['x2']],
            [c['y1'], c['y2']],
            color='red',
            linestyle='-',
            linewidth=2
        )
        plt.annotate(
            '({:.2f}, {:.2f})'.format(c['x1'], c['y1']),
            xy=(c['x1'] + 3, c['y1']),
            color='red'
        )
        plt.annotate(
            '({:.2f}, {:.2f})'.format(c['x2'], c['y2']),
            xy=(c['x2'], c['y2'] - 1500),
            color='red'
        )

        plt.xlabel('Accelerometer Reading (g)')
        plt.ylabel('RUL (s)')
        title = 'Bearing_1_1_acc_linear_regression_model_{0}'.format(c['title'])
        plt.title(title)
        plt.legend(
            (lineAcc, lineModel),
            ('absolute value of hacc', 'linear regression model'),
            loc='upper right'
        )
        if not os.path.exists(DEST_DIR):
            os.makedirs(DEST_DIR)
        plt.savefig(
            os.path.join(DEST_DIR, '{0}.jpg'.format(title)),
            dpi=400,
            format='jpg'
        )
        plt.clf()

main()
