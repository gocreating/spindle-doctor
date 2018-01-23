"""
Usage:
set KERAS_BACKEND=tensorflow python keras-anomaly-detection.py \
    --srcs \
        "../build/data/phm2012/feature-extracted/Learning_set-Bearing1_1-acc.csv" \
        "../build/data/phm2012/feature-extracted/Learning_set-Bearing1_2-acc.csv" \
    --columns \
        "avg" "max" "min" "anomaly" \
    --name keras-ad-phm \
    --step-size 16 \
    --input-size 3 \
    --hidden-size 16 \
    --output-size 2 \
    --batch-size 128 \
    --layer-depth 1 \
    --dropout-rate 0.1 \
    --learning-rates \
        1   1000   0.001 \
    --sample-size 100
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
import tensorflow as tf
from utils.utils import log, get_args, prepare_directory, get_batch_count
from utils.preprocess import unison_shuffled_copies, get_windowed_data
from models.EncDec import Model
from keras.models import Sequential  # 按順序建立的層
from keras.layers import Dense, LSTM, Activation      # 全連接層
# from keras.layers.core import Dense, AutoEncoder, TimeDistributedDense, Activation
from keras.optimizers import RMSprop

mpl.rcParams['agg.path.chunksize'] = 10000

# network parameters
# N_THREADS = 16

Y_LIMIT = [0, 2]

args = get_args()

# to be caculated
dataset = {
    'train': np.empty((0, args.step_size, len(args.columns) - 1)),
    'validate': np.empty((0, args.step_size, len(args.columns) - 1)),
    'anomalous': np.empty((0, args.step_size, len(args.columns) - 1)),
}

def read_dataset():
    global dataset

    for src, in zip(args.srcs):
        df = pd.read_csv(src, usecols=args.columns)
        df_normal = df[df['anomaly'] == 0].drop('anomaly', 1)
        df_anomalous = df[df['anomaly'] == 1].drop('anomaly', 1)
        normal_data = get_windowed_data(df_normal.values, args.step_size)
        anomalous_data = get_windowed_data(df_anomalous.values, args.step_size)

        np.random.shuffle(normal_data)
        train_data, validate_data = np.split(normal_data, [int(.9 * len(normal_data))])

        dataset['train'] = np.concatenate((dataset['train'], train_data), axis=0)
        dataset['validate'] = np.concatenate((dataset['validate'], validate_data), axis=0)
        dataset['anomalous'] = np.concatenate((dataset['anomalous'], anomalous_data), axis=0)

    # for src, in zip(args.srcs):
    #     df = pd.read_csv(src, usecols=args.columns)
    #     df_normal = df[df['anomaly'] == 0].drop('anomaly', 1)
    #     df_anomalous = df[df['anomaly'] == 1].drop('anomaly', 1)
    #     normal_data = get_windowed_data(df_normal[args.columns[0]].values, args.step_size)
    #     anomalous_data = get_windowed_data(df_anomalous[args.columns[0]]..values, args.step_size)
    #     np.random.shuffle(normal_data)
    #     train_data, validate_data = np.split(normal_data, [int(.9 * len(normal_data))])
    #
    #     dataset['train'] = np.concatenate((dataset['train'], train_data), axis=0)
    #     dataset['validate'] = np.concatenate((dataset['validate'], validate_data), axis=0)
    #     dataset['anomalous'] = np.concatenate((dataset['anomalous'], anomalous_data), axis=0)

def eval_mse(model, sess, dataset_name):
    mse = 0.0
    batch_count, data_size = get_batch_count(dataset[dataset_name], args.batch_size)
    for batch_idx in range(0, batch_count):
        begin_idx = batch_idx * args.batch_size
        end_idx = min(begin_idx + args.batch_size, data_size)
        mse = mse + model.error.eval(session=sess, feed_dict={
            model.xs: dataset[dataset_name][begin_idx: end_idx],
            model.ys: dataset[dataset_name][begin_idx: end_idx],
        })
    return mse / batch_count

def visualize_dataset(model, sess, epoch, dataset_name):
    dest_dir = prepare_directory(os.path.join(
        '../build/plots',
        args.scope,
        args.name
    ))

    if args.sample_size:
        x_axis = np.linspace(
            0,
            len(dataset[dataset_name]) - 1,
            num=args.sample_size,
            dtype=int
        )
        ground_truth = dataset[dataset_name][x_axis, 0, 0]
        ps = model.prediction.eval(
            session=sess,
            feed_dict={
                model.xs: dataset[dataset_name][x_axis],
                model.ys: dataset[dataset_name][x_axis],
            }
        )
        predicted = np.array(ps)[:, 0, 0]

    plt.ylim(Y_LIMIT)
    plt.scatter(x_axis, ground_truth, color='green', marker='x', s=12)
    plt.scatter(x_axis, predicted, color='blue', s=10, linewidth=0)
    plt.plot(x_axis, np.absolute(predicted - ground_truth), color='red', linestyle='--', linewidth=1)

    mse = eval_mse(model, sess, dataset_name)

    title = '{0}\nepoch-{1}\nmse = {2}'.format(dataset_name, epoch, mse)
    plt.title(title)
    plt.savefig(
        os.path.join(dest_dir, 'epoch-{0}-{1}.png'.format(epoch, dataset_name)),
        dpi=400,
        format='png'
    )
    plt.clf()

    return mse

if __name__ == '__main__':
    read_dataset()

    m = Sequential()
    m.add(LSTM(args.hidden_size, input_dim=args.input_size, return_sequences=True))
    m.add(LSTM(args.input_size, return_sequences=True))
    m.add(Activation('linear'))
    m.compile(loss='mse', optimizer='RMSprop')
    # m.fit(dataset['train'], dataset['train'], nb_epoch=200, batch_size=args.batch_size)
    m.fit(dataset['train'], dataset['train'], nb_epoch=200)
