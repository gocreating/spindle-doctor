# -*- coding: utf-8 -*-
"""
Usage:
python test-classification-anomaly-detection-for-edge.py ^
    --columns "avg" "level_normalized_fft1" "anomaly" ^
    --column "level_normalized_fft1" ^
    --threshold 0.000403317078173349 ^
    --step-size 32 ^
    --hidden-size 64 ^
    --embedding-size 128 ^
    --symbol-size 8 ^
    --batch-size 128 ^
    --layer-depth 2 ^
    --dropout-rate 0.1 ^
    --src "../build/models/edge-server/deployed-model/model" ^
    --test-src "../build/data/phm2012/feature-8-level-extracted/Learning_set-Bearing1_1-acc.csv"
"""
import sys
import os
import time
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from utils.utils import log, get_args, get_batch_count
from utils.preprocess import get_windowed_data
from models.EncDecEmbedding import Model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
args = get_args()
dataset = np.empty((0, args.step_size))

def read_dataset():
    global dataset

    df = pd.read_csv(args.test_src, usecols=args.columns)
    feature_data = np.reshape(get_windowed_data(df[args.column].values, args.step_size), (-1, args.step_size))
    dataset = np.concatenate((dataset, feature_data), axis=0)

    # cut last batch
    length = len(dataset)
    r = length % args.batch_size
    if r != 0:
        dataset = dataset[0:-r]

if __name__ == '__main__':
    read_dataset()
    tf.reset_default_graph()
    with tf.Graph().as_default():
        tf.set_random_seed(args.seed)
        sess = tf.InteractiveSession()
        importSaver = tf.train.import_meta_graph(os.path.join(args.src, '../model.meta'))
        importSaver.restore(sess, args.src)

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
            is_anomaly = mse[-1] > args.threshold
            print('Anomaly' if is_anomaly else 'Normal')
