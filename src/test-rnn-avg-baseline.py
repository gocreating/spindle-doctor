"""
Usage:
python test-rnn-avg-baseline.py \
    --columns "avg" "fft1" "anomaly" \
    --column "fft1" \
    --title "Anomaly Detection Baseline" \
    --threshold 857 \
    --scope phm2012 \
    --name test-phm-rnn-avg-baseline \
    --step-size 32 \
    --sample-size 256 \
    --test-src "../build/data/phm2012/feature-extracted/Learning_set-Bearing1_1-acc.csv"
"""
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from utils.utils import log, get_args, prepare_directory, get_batch_count

args = get_args()
df = None
xs = []
ys = []

def visualize(mses):
    dest_dir = prepare_directory(os.path.join(
        '../build/plots',
        args.scope,
        args.name,
        os.path.basename(args.test_src).rsplit('.', 1)[0]
    ))

    f, axarr = plt.subplots(2, sharex=True)

    axarr[0].set_title(args.title)
    axarr[0].set_ylabel('Vibration Signal (g)')
    axarr[1].set_ylabel('Error (MSE)')
    plt.xlabel('Bearing Life (390ms)')

    threshold = args.threshold
    anomaly_flags = mses >= threshold
    colors = ['red' if a else 'green' for a in anomaly_flags]
    lines = [((x0,y0), (x1,y1)) for x0, y0, x1, y1 in zip(xs[:-1], ys[:-1], xs[1:], ys[1:])]
    colored_lines = LineCollection(lines, colors=colors, linewidths=(1,))
    axarr[0].add_collection(colored_lines)
    axarr[0].autoscale_view()
    axarr[0].plot([], [], c='green', label='predicted normal')
    axarr[0].plot([], [], c='red', label='predicted anomalous')
    axarr[0].legend()

    axarr[1].plot(xs, mses, color='blue', label='reconstruct error')
    axarr[1].plot([xs[0], xs[-1]], [threshold, threshold], color='blue', linestyle='--', linewidth=1, label='anomaly threshold')
    axarr[1].legend()

    plt.savefig(
        os.path.join(dest_dir, '{0}.png'.format(args.name)),
        dpi=400,
        format='png'
    )
    plt.clf()

def eval_ROC(mses):
    df['mse'] = mses
    df_normal = df[df['anomaly'] == 0].drop('anomaly', 1)
    df_anomalous = df[df['anomaly'] == 1].drop('anomaly', 1)

    df_TP = df_anomalous[df_anomalous['mse'] >= args.threshold]
    df_FP = df_normal[df_normal['mse'] >= args.threshold]
    df_TN = df_normal[df_normal['mse'] < args.threshold]
    df_FN = df_anomalous[df_anomalous['mse'] < args.threshold]

    TP, FP, TN, FN = len(df_TP), len(df_FP), len(df_TN), len(df_FN)
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    print('TP =\t{0}\nFP =\t{1}\nTN =\t{2}\nFN =\t{3}\nTPR =\t{4}\nFPR =\t{5}'.format(TP, FP, TN, FN, TPR, FPR))

if __name__ == '__main__':
    df = pd.read_csv(args.test_src, usecols=args.columns)
    df['predicted'] = df[args.column].rolling(args.step_size).mean().shift(1)
    df = df[args.step_size:]
    mses = (df['predicted'] - df[args.column]) ** 2
    eval_ROC(mses)
    # df['mse'] = (df['predicted'] - df[args.column]) ** 2
    #
    # # Compute ROC
    # df_normal = df[df['anomaly'] == 0].drop('anomaly', 1)
    # df_anomalous = df[df['anomaly'] == 1].drop('anomaly', 1)
    #
    # df_TP = df_anomalous[df_anomalous['mse'] >= args.threshold]
    # df_FP = df_normal[df_normal['mse'] >= args.threshold]
    # df_TN = df_normal[df_normal['mse'] < args.threshold]
    # df_FN = df_anomalous[df_anomalous['mse'] < args.threshold]
    #
    # TP, FP, TN, FN = len(df_TP), len(df_FP), len(df_TN), len(df_FN)
    # TPR = TP / (TP + FN)
    # FPR = FP / (FP + TN)
    # print('TP =\t{0}\nFP =\t{1}\nTN =\t{2}\nFN =\t{3}\nTPR =\t{4}\nFPR =\t{5}'.format(TP, FP, TN, FN, TPR, FPR))

    xs = np.linspace(
        0,
        len(df) - 1,
        num=args.sample_size,
        dtype=int
    )
    ys = np.array(df[args.columns[0]].values)[xs]
    mses = np.array(df['mse'].values)[xs]
    visualize(mses)
