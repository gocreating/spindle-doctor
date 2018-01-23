"""
Usage:
python regression-anomaly-detection.py \
    --srcs \
        "../build/data/phm2012/feature-level-extracted/Learning_set-Bearing1_1-acc.csv" \
        "../build/data/phm2012/feature-level-extracted/Learning_set-Bearing1_2-acc.csv" \
    --columns \
        "fft1" "anomaly" \
    --scope phm2012 \
    --name phm-fft1-regression \
    --step-size 32 \
    --input-size 1 \
    --hidden-size 64 \
    --output-size 1 \
    --batch-size 128 \
    --layer-depth 1 \
    --dropout-rate 0.1 \
    --learning-rates \
        1 500 0.001 \
    --sample-size 128 \
    --src ../build/models/phm2012/phm-fft1-regression/model \
    --dest ../build/models/phm2012/phm-fft1-regression/model
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
from models.EncDec import Model

mpl.rcParams['agg.path.chunksize'] = 10000

# network parameters
# N_THREADS = 16

Y_LIMIT = [0, 1]

args = get_args()

# to be caculated
dataset = {
    'train': np.empty((0, args.step_size, len(args.columns) - 1)),
    'validate': np.empty((0, args.step_size, len(args.columns) - 1)),
    'anomalous': np.empty((0, args.step_size, len(args.columns) - 1)),
}

def read_dataset():
    global dataset
    # values = np.array([])

    for src, in zip(args.srcs):
        df = pd.read_csv(src, usecols=args.columns)
        # values = np.concatenate((values, df[args.columns[0]].values), axis=0)
        df_normal = df[df['anomaly'] == 0].drop('anomaly', 1)
        df_anomalous = df[df['anomaly'] == 1].drop('anomaly', 1)
        normal_data = get_windowed_data(df_normal.values, args.step_size)
        anomalous_data = get_windowed_data(df_anomalous.values, args.step_size)

        np.random.shuffle(normal_data)
        train_data, validate_data = np.split(normal_data, [int(.9 * len(normal_data))])

        dataset['train'] = np.concatenate((dataset['train'], train_data), axis=0)
        dataset['validate'] = np.concatenate((dataset['validate'], validate_data), axis=0)
        dataset['anomalous'] = np.concatenate((dataset['anomalous'], anomalous_data), axis=0)

    # minimum = np.amin(values)
    # maximum = np.amax(values)

    # for name in ['train', 'validate', 'anomalous']:
    #     dataset[name] = (dataset[name] - minimum) / (maximum - minimum)

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
    # step = 4, hidden size = 128, lr = 0.00001
    # tf.set_random_seed(123456) # 0.244
    # tf.set_random_seed(9487) # 0.452
    # tf.set_random_seed(999) # 0.538
    # tf.set_random_seed(9999) # 0.129

    # step = 16, hidden size = 128
    # tf.set_random_seed(123456) # 0.45
    # tf.set_random_seed(9487) # 0.494
    # tf.set_random_seed(94879487) # 0.365
    # tf.set_random_seed(948794879487) # 0.436
    # tf.set_random_seed(99) # 0.466
    # tf.set_random_seed(999) # 0.455
    # tf.set_random_seed(9999) # 0.517
    # tf.set_random_seed(99999) # 0.603
    # tf.set_random_seed(999999) # 0.427
    # tf.set_random_seed(9999999) # 0.38
    # tf.set_random_seed(99999999) # 0.391
    # tf.set_random_seed(0) # 0.318
    # tf.set_random_seed(1) # 0.381
    # tf.set_random_seed(2) # 0.351
    # tf.set_random_seed(3) # 0.303
    # tf.set_random_seed(4) # 0.439
    # tf.set_random_seed(5) # 0.595
    # tf.set_random_seed(6) # 0.479
    # tf.set_random_seed(7) # 0.481
    # tf.set_random_seed(8) # 0.668
    # tf.set_random_seed(9) # 0.468
    # tf.set_random_seed(10) # 0.512
    # tf.set_random_seed(11) # 0.502
    # tf.set_random_seed(515415) # 0.507
    # tf.set_random_seed(987456) # 0.208
    # tf.set_random_seed(42158989) # 0.495
    # tf.set_random_seed(987457) # 0.275
    # tf.set_random_seed(987455) # 0.611
    # tf.set_random_seed(532184697) # 0.38
    # tf.set_random_seed(786456) # 0.325
    # tf.set_random_seed(78644545654345) # 0.326
    # tf.set_random_seed(48564561) # 0.372
    # tf.set_random_seed(86789724455) # 0.329
    # tf.set_random_seed(952874321) # 0.294

    # step = 32, hidden size = 64
    # tf.set_random_seed(123456) # 0.00356
    tf.set_random_seed(args.seed) # 0.00225
    # tf.set_random_seed(999) # 0.00323

    with tf.variable_scope('anomaly-detection', reuse=None):
        model = Model(
            args.step_size,
            args.input_size,
            args.hidden_size,
            args.output_size,
            args.layer_depth,
            args.batch_size,
            args.dropout_rate,
            False
        )
    with tf.variable_scope('anomaly-detection', reuse=True):
        model_production = Model(
            args.step_size,
            args.input_size,
            args.hidden_size,
            args.output_size,
            args.layer_depth,
            args.batch_size,
            args.dropout_rate,
            True
        )

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
        validate_mse = visualize_dataset(model_production, sess, 0, 'validate')
        anomalous_mse = visualize_dataset(model_production, sess, 0, 'anomalous')
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
                            model.learning_rate: learning_rate,
                        }
                    )

                    if (batch_idx + 1) % 1000 == 0:
                        elapsed_time = time.time() - start_time
                        print('Epoch\t%d, Batch\t%d, Elapsed time\t%.1fs' % (
                            epoch, batch_idx + 1, elapsed_time
                        ))

                elapsed_time = time.time() - start_time
                validate_mse = visualize_dataset(model_production, sess, epoch, 'validate')
                anomalous_mse = visualize_dataset(model_production, sess, epoch, 'anomalous')
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
