"""
Usage:
python activation-anomaly-detection.py \
    --srcs \
        "../build/data/phm2012/feature-level-extracted/Learning_set-Bearing1_1-acc.csv" \
        "../build/data/phm2012/feature-level-extracted/Learning_set-Bearing1_2-acc.csv" \
    --columns \
        "level_fft1" "anomaly" \
    --scope phm2012 \
    --name phm-fft1-classification \
    --step-size 16 \
    --hidden-size 16 \
    --embedding-size 128 \
    --symbol-size 128 \
    --batch-size 128 \
    --layer-depth 1 \
    --dropout-rate 0.1 \
    --learning-rates \
        1 100 0.1 \
        101 200 0.01 \
        201 1000 0.001 \
    --sample-size 128 # must >= batch_size and will be cut to match batch_size \
    --src-breakpoint "../build/meta/phm2012/anomaly-detection-unequal/breakpoint-128.csv" \
    --src ../build/models/phm2012/phm-fft1-classification/model \
    --dest ../build/models/phm2012/phm-fft1-classification/model
"""
import sys
import os
import time
import math
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from utils.utils import log, get_args, prepare_directory, get_batch_count
from utils.preprocess import get_windowed_data
from models.CustomActivation import Model

mpl.rcParams['agg.path.chunksize'] = 10000

# network parameters
# N_THREADS = 16

args = get_args()

# to be caculated
dataset = {
    'train': np.empty((0, args.step_size)),
    'validate': np.empty((0, args.step_size)),
    'anomalous': np.empty((0, args.step_size)),
}

Y_LIMIT = [0, args.symbol_size + 1]

def read_dataset():
    global dataset

    for src, in zip(args.srcs):
        df = pd.read_csv(src, usecols=args.columns)
        df_normal = df[df['anomaly'] == 0].drop('anomaly', 1)
        df_anomalous = df[df['anomaly'] == 1].drop('anomaly', 1)

        normal_data = np.reshape(get_windowed_data(df_normal.values, args.step_size), (-1, args.step_size))
        anomalous_data = np.reshape(get_windowed_data(df_anomalous.values, args.step_size), (-1, args.step_size))

        if not args.no_shuffle:
            np.random.shuffle(normal_data)
        train_data, validate_data = np.split(normal_data, [int(.9 * len(normal_data))])

        dataset['train'] = np.concatenate((dataset['train'], train_data), axis=0)
        dataset['validate'] = np.concatenate((dataset['validate'], validate_data), axis=0)
        dataset['anomalous'] = np.concatenate((dataset['anomalous'], anomalous_data), axis=0)

    for name in ['train', 'validate', 'anomalous']:
        # cut last batch
        d = dataset[name]
        length = len(d)
        r = length % args.batch_size
        if r != 0:
            dataset[name] = dataset[name][0:-r]

def get_mse_weights():
    if args.src_breakpoint:
        df_breakpoints = pd.read_csv(args.src_breakpoint)
        breakpoints = df_breakpoints[args.columns[0].replace('level_', '')].values
        mse_weights = np.concatenate((
            [breakpoints[0]],
            np.array([(breakpoints[i] + breakpoints[i + 1]) / 2 for i in range(0, len(breakpoints) - 1)]),
            [breakpoints[-1]]
        ))
        return mse_weights
    elif args.src_centroid:
        df_centroids = pd.read_csv(args.src_centroid)
        centroids = df_centroids[args.columns[0].replace('klevel_', '')].values
        return centroids

def eval_mse(model, sess, dataset_name):
    mse = 0.0
    batch_count, data_size = get_batch_count(dataset[dataset_name], args.batch_size)
    for batch_idx in range(0, batch_count):
        begin_idx = batch_idx * args.batch_size
        end_idx = min(begin_idx + args.batch_size, data_size)
        mse = mse + model.mse_error.eval(session=sess, feed_dict={
            model.xs: dataset[dataset_name][begin_idx: end_idx],
            model.ys: dataset[dataset_name][begin_idx: end_idx],
            model.feed_previous: True,
        })
    return mse / batch_count

def visualize_dataset(model, sess, epoch, dataset_name):
    dest_dir = prepare_directory(os.path.join(
        '../build/plots',
        args.scope,
        args.name
    ))

    if args.sample_size:
        args.sample_size = args.sample_size - (args.sample_size % args.batch_size)
        x_axis = np.linspace(
            0,
            len(dataset[dataset_name]) - 1,
            num=args.sample_size,
            dtype=int
        )
        ground_truth = dataset[dataset_name][x_axis, 0]
        ps = model.prediction.eval(
            session=sess,
            feed_dict={
                model.xs: dataset[dataset_name][x_axis],
                model.ys: dataset[dataset_name][x_axis],
                model.feed_previous: True,
            }
        )
        predicted = np.array(ps)[:, 0]

    plt.ylim(Y_LIMIT)
    plt.scatter(x_axis, ground_truth, color='green', marker='x', s=12)
    plt.scatter(x_axis, predicted, color='blue', s=10, linewidth=0)
    plt.plot(x_axis, abs(predicted - ground_truth), color='red', linestyle='--', linewidth=1)

    mse = eval_mse(model, sess, dataset_name)

    title = 'epoch-{0}\n{1} mse = {2}'.format(epoch, dataset_name, mse)
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
        args.rnn_unit,
        mse_weights
    )

    # parameter_size = np.sum([np.prod(v.shape) for v in tf.trainable_variables()])
    # print(parameter_size)
    # sys.exit(0)

    # start session
    sess = tf.InteractiveSession(
        # config=tf.ConfigProto(intra_op_parallelism_threads=N_THREADS)
    )

    # prepare model import or export
    if args.src:
        importSaver = tf.train.Saver()
        importSaver.restore(sess, args.src)
    else:
        # initize variable
        sess.run(tf.global_variables_initializer())

    if args.dest:
        exportSaver = tf.train.Saver()
        prepare_directory(os.path.dirname(args.dest))

    filename = args.log or os.path.join(
        prepare_directory(os.path.join(
            '../build/plots', args.scope, args.name)
        ), 'log.csv'
    )
    min_validate_mse = 999999
    batch_count, data_size = get_batch_count(dataset['train'], args.batch_size)
    with open(filename, 'w') as fd_log:
        start_time = time.time()

        # before training
        validate_mse = visualize_dataset(model, sess, 0, 'validate')
        anomalous_mse = visualize_dataset(model, sess, 0, 'anomalous')
        print('Epoch\t%d, Batch\t%d, Elapsed time\t%.1fs, Validate MSE\t%s, Anomalous MSE\t%s, Min Validate MSE\t%s' % (
            0, 0, 0, validate_mse, anomalous_mse, min_validate_mse
        ))

        learning_rates_schedules = np.reshape(
            args.learning_rates,
            (-1, 3)
        )
        for schedule in learning_rates_schedules:
            learning_rate = schedule[2]

            # loop epochs
            for epoch in range(int(schedule[0]), int(schedule[1]) + 1):
                for batch_idx in range(0, batch_count):
                    begin_idx = batch_idx * args.batch_size
                    end_idx = min(begin_idx + args.batch_size, data_size)

                    model.train_step.run(
                        feed_dict={
                            model.xs: dataset['train'][begin_idx: end_idx],
                            model.ys: dataset['train'][begin_idx: end_idx],
                            model.feed_previous: False,
                            model.learning_rate: learning_rate,
                        }
                    )

                    _, gradients, sg0 = sess.run([model.train_step, model.gradients, model.sg0], feed_dict={
                        model.xs: dataset['train'][begin_idx: end_idx],
                        model.ys: dataset['train'][begin_idx: end_idx],
                        model.feed_previous: False,
                        model.learning_rate: learning_rate,
                    })
                    if batch_idx is 0:
                        print('=====================')
                        # print(gradients[0])
                        # print('=====================')
                        # print(g0)
                        # print(tf.convert_to_tensor(gradients[0])) # not working
                        # a = np.array(g0)
                        # b = np.array(gradients[0].values)
                        # print(a)
                        # print(a.shape)
                        # print('')
                        # print(b)
                        # print(b.shape)
                        print(sg0)
                        print('=====================')

                    if (batch_idx + 1) % 1000 == 0:
                        elapsed_time = time.time() - start_time
                        print('Epoch\t%d, Batch\t%d, Elapsed time\t%.1fs' % (
                            epoch, batch_idx + 1, elapsed_time
                        ))

                elapsed_time = time.time() - start_time
                validate_mse = visualize_dataset(model, sess, epoch, 'validate')
                anomalous_mse = visualize_dataset(model, sess, epoch, 'anomalous')
                if validate_mse < min_validate_mse:
                    min_validate_mse = validate_mse
                    if args.dest:
                        exportSaver.save(sess, args.dest)
                print('Epoch\t%d, Batch\t%d, Elapsed time\t%.1fs, Validate MSE\t%s, Anomalous MSE\t%s, Min Validate MSE\t%s' % (
                    epoch, batch_count, elapsed_time, validate_mse, anomalous_mse, min_validate_mse
                ))
                fd_log.write('{0},{1},{2},{3}\n'.format(
                    epoch, validate_mse, anomalous_mse, elapsed_time
                ))
                fd_log.flush()
