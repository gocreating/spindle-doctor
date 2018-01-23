"""
Usage:
python test-classification-anomaly-detection.py \
    --columns "avg" "level_normalized_fft1" "anomaly" \
    --column "level_normalized_fft1" \
    --title "Anomaly Detection by Unequal Segmentation" \
    --threshold 0.000849950254676382 \
    --scope phm2012 \
    --name test-phm-normalized-fft1-classification-128 \
    --step-size 32 \
    --hidden-size 64 \
    --embedding-size 128 \
    --symbol-size 128 \
    --batch-size 128 \
    --layer-depth 2 \
    --dropout-rate 0.1 \
    --src-breakpoint "../build/meta/phm2012/anomaly-detection-unequal/breakpoint-128.csv" \
    --sample-size 256 \
    --src "../build/models/phm2012/phm-normalized-fft1-classification-128/model" \
    --test-src "../build/data/phm2012/feature-128-level-extracted/Learning_set-Bearing1_1-acc.csv" \
    --batch-step 1
"""
import sys
import os
import time
import math
import numpy as np
import pandas as pd
import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import tensorflow as tf
from utils.utils import log, get_args, prepare_directory, get_batch_count, smooth
from utils.preprocess import get_windowed_data
from models.EncDecEmbedding import Model

mpl.rcParams['agg.path.chunksize'] = 10000

args = get_args()

# to be caculated
df = None
xs = []
ys = []
dataset = {
    'ordered': np.empty((0, args.step_size)),
    'sampled': np.empty((0, args.step_size)),
}
def read_dataset():
    global df
    global xs
    global ys
    global dataset

    df = pd.read_csv(args.test_src, usecols=args.columns)
    normal_data = np.reshape(get_windowed_data(df[args.column].values, args.step_size), (-1, args.step_size))
    dataset['ordered'] = np.concatenate((dataset['ordered'], normal_data), axis=0)

    xs = np.linspace(
        0,
        len(dataset['ordered']) - 1,
        num=args.sample_size - args.sample_size % args.batch_size,
        dtype=int
    )
    ys = np.array(df[args.columns[0]].values)[xs]
    dataset['sampled'] = dataset['ordered'][xs]

    for name in ['ordered']:
        # cut last batch
        d = dataset[name]
        length = len(d)
        r = length % args.batch_size
        if r != 0:
            dataset[name] = dataset[name][0:-r]
    df = df[0: len(dataset['ordered'])]

def get_mse_weights():
    df_breakpoints = pd.read_csv(args.src_breakpoint)
    breakpoints = df_breakpoints[args.column.replace('level_', '')].values
    mse_weights = np.concatenate((
        [breakpoints[0]],
        np.array([(breakpoints[i] + breakpoints[i + 1]) / 2 for i in range(0, len(breakpoints) - 1)]),
        [breakpoints[-1]]
    ))
    return mse_weights

# https://stackoverflow.com/questions/37787632/different-color-for-line-depending-on-corresponding-values-in-pyplot
def visualize(mses):
    dest_dir = prepare_directory(os.path.join(
        '../build/plots',
        args.scope,
        args.name,
        os.path.basename(args.test_src).rsplit('.', 1)[0]
    ))

    f, axarr = plt.subplots(2, sharex=True)

    axarr[0].set_title(args.title)
    axarr[0].set_ylabel('Vibration Signal')
    axarr[1].set_ylabel('Reconstruct Error (MSE)')
    plt.xlabel('Bearing Life')

    threshold = args.threshold
    anomaly_flags = mses >= threshold
    colors = ['red' if a else 'green' for a in anomaly_flags]
    lines = [((x0,y0), (x1,y1)) for x0, y0, x1, y1 in zip(xs[:-1], ys[:-1], xs[1:], ys[1:])]
    colored_lines = LineCollection(lines, colors=colors, linewidths=(1,))
    axarr[0].add_collection(colored_lines)
    axarr[0].autoscale_view()
    axarr[0].plot([], [], c='green', label='normal')
    axarr[0].plot([], [], c='red', label='anomalous')
    bound = xs[int(len(xs) * 0.9)]
    axarr[0].plot([bound, bound], [np.amin(ys), np.amax(ys)], color='blue', linestyle='--', linewidth=1)
    axarr[0].legend()

    axarr[1].plot(xs, mses, color='blue', label='reconstruct error')
    axarr[1].plot([xs[0], xs[-1]], [threshold, threshold], color='blue', linestyle='--', linewidth=1, label='anomaly threshold')
    axarr[1].legend()

    plt.savefig(
        os.path.join(dest_dir, '{0}(seed={1}, smooth={2}).eps'.format(args.name, args.seed, args.smooth)),
        dpi=800,
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
    read_dataset()

    tf.reset_default_graph()
    with tf.Graph().as_default():

        tf.set_random_seed(args.seed)
        mse_weights = get_mse_weights()
        model = Model(
            args.step_size,
            args.hidden_size,
            args.embedding_size,
            args.symbol_size,
            args.layer_depth,
            args.batch_size,
            args.dropout_rate,
            mse_weights
        )

        # start session
        sess = tf.InteractiveSession(
            # config=tf.ConfigProto(intra_op_parallelism_threads=N_THREADS)
        )

        # prepare model import or export
        importSaver = tf.train.Saver()
        importSaver.restore(sess, args.src)

        for context in ['sampled', 'ordered']:
            batch_count, data_size = get_batch_count(dataset[context], args.batch_size)
            mses = np.array([])
            for batch_idx in range(0, batch_count, args.batch_step):
                if context == 'ordered' and batch_idx % 50 == 0:
                    print('{0} / {1} batches'.format(batch_idx, batch_count))
                begin_idx = batch_idx * args.batch_size
                end_idx = min(begin_idx + args.batch_size, data_size)
                ground_truth = dataset[context][begin_idx: end_idx]
                restored_predictions = model.restored_prediction.eval(
                    session=sess,
                    feed_dict={
                        model.xs: ground_truth,
                        model.ys: ground_truth,
                        model.feed_previous: True,
                    }
                )
                restored_ys = model.restored_ys.eval(
                    session=sess,
                    feed_dict={
                        model.xs: ground_truth,
                        model.ys: ground_truth,
                        model.feed_previous: True,
                    }
                )
                # mse的計算方式會嚴重影響測試結果
                mse = np.mean((restored_ys - restored_predictions) ** 2, axis=1)
                # mse = np.reshape((restored_ys - restored_predictions) ** 2, (-1, args.step_size))[:, 0]
                mses = np.concatenate((mses, mse), axis=0)
            # smoothing
            if args.smooth:
                mses = smooth(mses, args.smooth)
            if context == 'sampled':
                visualize(mses)
            if context == 'ordered':
                eval_ROC(mses)
