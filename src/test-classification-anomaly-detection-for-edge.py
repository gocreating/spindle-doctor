# -*- coding: utf-8 -*-
"""
Usage:
python test-classification-anomaly-detection-for-edge.py ^
    --threshold 0.016 ^
    --step-size 32 ^
    --hidden-size 64 ^
    --embedding-size 128 ^
    --symbol-size 256 ^
    --batch-size 128 ^
    --layer-depth 2 ^
    --dropout-rate 0.1 ^
    --src "../build/models/edge-server/deployed-model/model" ^
    --test-src "../build/data/edge-server/vibration.csv" ^
    --src-breakpoint "../build/meta/phm2012/breakpoints-from-feature/breakpoint-256.csv"

"""
import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from bisect import bisect_left
from utils.utils import get_args, prepare_directory
from utils.preprocess import get_windowed_data
from utils.input import get_batch
import importlib

p = importlib.import_module('extract-feature')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
args = get_args()

def read_dataset():
    dataset = np.empty((0, args.step_size))
    df_chunks = pd.read_csv(
        args.test_src,
        header=None,
        names=['x'],
        chunksize=100000
    )

    # feature extration
    ffts = []
    for batch_idx, df_batch in get_batch(df_chunks, args.batch_size):
        values = np.array(df_batch['x'])
        fft1, fft2 = p.get_fft(values, args.batch_size)
        ffts.append(fft1)
    ffts = np.array(ffts)
    minimum = np.amin(ffts)
    maximum = np.amax(ffts)
    normalized_ffts = (ffts - minimum) / (maximum - minimum)

    # quantization
    df_breakpoints = pd.read_csv(args.src_breakpoint)
    level_normalized_ffts = np.array([
        bisect_left(df_breakpoints['normalized_fft1'], element)
        for element in normalized_ffts
    ])

    # unfold
    feature_data = np.reshape(get_windowed_data(level_normalized_ffts, args.step_size), (-1, args.step_size))
    dataset = np.concatenate((dataset, feature_data), axis=0)

    # cut last batch
    length = len(dataset)
    r = length % args.batch_size
    if r != 0:
        dataset = dataset[0:-r]

    return dataset

if __name__ == '__main__':
    tf.reset_default_graph()
    with tf.Graph().as_default():
        tf.set_random_seed(args.seed)
        sess = tf.InteractiveSession()
        importSaver = tf.train.import_meta_graph(os.path.join(args.src, '../model.meta'))
        importSaver.restore(sess, args.src)

        while True:
            dataset = read_dataset()
            for batch_idx in range(0, 1):
                begin_idx = batch_idx * args.batch_size
                end_idx = begin_idx + args.batch_size
                xs = dataset[begin_idx: end_idx]

                restored_predictions = sess.run('compute_cost/Reshape_1:0', feed_dict={
                    'input_layer/xs:0': xs,
                    'input_layer/ys:0': xs,
                    'input_layer/feed_previous:0': True,
                })
                restored_ys = sess.run('compute_cost/Reshape_3:0', feed_dict={
                    'input_layer/xs:0': xs,
                    'input_layer/ys:0': xs,
                    'input_layer/feed_previous:0': True,
                })
                mse = np.mean((restored_ys - restored_predictions) ** 2, axis=1)
                is_anomaly = mse[0] > args.threshold

                dest_dir = prepare_directory(os.path.join(
                    args.src,
                    '../../inference-result'
                ))
                with open(os.path.join(dest_dir, 'last.txt'), 'w') as fd_result:
                    print('anomaly' if is_anomaly else 'normal')
                    fd_result.write('anomaly' if is_anomaly else 'normal')
            time.sleep(5)
