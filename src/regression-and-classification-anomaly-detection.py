"""
Usage:
python regression-and-classification-anomaly-detection.py
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
from utils.preprocess import get_windowed_data, unison_shuffled_copies
from models.Linear_Regression_And_Classification import Model

mpl.rcParams['agg.path.chunksize'] = 10000

# network parameters
# N_THREADS = 16

args = get_args()

# to be caculated
dataset = {
    'train_feature': np.empty((0, args.step_size, args.input_size)),
    'train_label': np.empty((0, 1)),
    'validate_feature': np.empty((0, args.step_size, args.input_size)),
    'validate_label': np.empty((0, 1)),
}

Y_LIMIT = [-1, args.symbol_size]

def read_dataset():
    global dataset

    for src, in zip(args.srcs):
        df = pd.read_csv(src, usecols=args.columns)
        df_feature = df.drop('anomaly', 1)
        df_label = df['anomaly']

        feature_data = np.reshape(get_windowed_data(df_feature.values, args.step_size), (-1, args.step_size, args.input_size))
        label_data = np.reshape(df_label.values, [-1, 1])[0:-(args.step_size - 1)]

        if not args.no_shuffle:
            shuffled_feature_data, shuffled_label_data = unison_shuffled_copies(
                feature_data, label_data
            )
        train_feature_data, validate_feature_data = np.split(shuffled_feature_data, [int(.9 * len(shuffled_feature_data))])
        train_label_data, validate_label_data = np.split(shuffled_label_data, [int(.9 * len(shuffled_label_data))])

        dataset['train_feature'] = np.concatenate((dataset['train_feature'], train_feature_data), axis=0)
        dataset['train_label'] = np.concatenate((dataset['train_label'], train_label_data), axis=0)
        dataset['validate_feature'] = np.concatenate((dataset['validate_feature'], validate_feature_data), axis=0)
        dataset['validate_label'] = np.concatenate((dataset['validate_label'], validate_label_data), axis=0)

    for name in ['train_feature', 'train_label', 'validate_feature', 'validate_label']:
        # cut last batch
        d = dataset[name]
        length = len(d)
        r = length % args.batch_size
        if r != 0:
            dataset[name] = dataset[name][0:-r]

def eval_metric(model, sess, dataset_name):
    acc = 0.0
    entropy = 0.0
    batch_count, data_size = get_batch_count(dataset[dataset_name + '_feature'], args.batch_size)
    for batch_idx in range(0, batch_count):
        begin_idx = batch_idx * args.batch_size
        end_idx = min(begin_idx + args.batch_size, data_size)
        _acc, _entropy = sess.run([model.accuracy, model.entropy], feed_dict={
            model.xs: dataset[dataset_name + '_feature'][begin_idx: end_idx],
            model.ys: dataset[dataset_name + '_label'][begin_idx: end_idx],
        })
        acc = acc + _acc
        entropy = entropy + _entropy
        # acc = acc + model.accuracy.eval(session=sess, feed_dict={
        #     model.xs: dataset[dataset_name + '_feature'][begin_idx: end_idx],
        #     model.ys: dataset[dataset_name + '_label'][begin_idx: end_idx],
        # })
    return acc / batch_count, entropy / batch_count

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
            len(dataset[dataset_name + '_feature']) - 1,
            num=args.sample_size,
            dtype=int
        )
        ground_truth = dataset[dataset_name + '_label'][x_axis, 0]
        ps = model.prediction.eval(
            session=sess,
            feed_dict={
                model.xs: dataset[dataset_name + '_feature'][x_axis],
                model.ys: dataset[dataset_name + '_label'][x_axis],
            }
        )
        predicted = np.array(ps)[:, 0]

    plt.ylim(Y_LIMIT)
    plt.scatter(x_axis, ground_truth, color='green', marker='x', s=12)
    plt.scatter(x_axis, predicted, color='blue', s=10, linewidth=0)
    plt.plot(x_axis, abs(predicted - ground_truth), color='red', linestyle='--', linewidth=1)

    acc, entropy = eval_metric(model, sess, dataset_name)

    title = 'epoch-{0}\n{1} accuracy = {2}'.format(epoch, dataset_name, acc)
    plt.title(title)
    plt.savefig(
        os.path.join(dest_dir, 'epoch-{0}-{1}.png'.format(epoch, dataset_name)),
        dpi=400,
        format='png'
    )
    plt.clf()

    return acc, entropy

if __name__ == '__main__':
    read_dataset()
    tf.set_random_seed(args.seed)
    model = Model(
        2,
        args.step_size,
        args.symbol_size,
        args.batch_size
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
    max_validate_acc = -999999
    min_validate_entropy = 999999
    batch_count, data_size = get_batch_count(dataset['train_feature'], args.batch_size)
    with open(filename, 'w') as fd_log:
        start_time = time.time()

        # before training
        validate_acc, validate_entropy = visualize_dataset(model, sess, 0, 'validate')
        # validate_acc = eval_acc(model, sess, 'validate')

        # anomalous_mse = visualize_dataset(model, sess, 0, 'anomalous')
        print('Epoch\t%d, Elapsed time\t%.1fs, Accuracy\t%s, Entropy\t%s, Max Accuracy\t%s, Min Entropy\t%s' % (
            0, 0, validate_acc, validate_entropy, max_validate_acc, min_validate_entropy
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
                            model.xs: dataset['train_feature'][begin_idx: end_idx],
                            model.ys: dataset['train_label'][begin_idx: end_idx],
                            model.learning_rate: learning_rate,
                        }
                    )

                    if (batch_idx + 1) % 1000 == 0:
                        elapsed_time = time.time() - start_time
                        print('Epoch\t%d, Batch\t%d, Elapsed time\t%.1fs' % (
                            epoch, batch_idx + 1, elapsed_time
                        ))

                elapsed_time = time.time() - start_time
                validate_acc, validate_entropy = visualize_dataset(model, sess, epoch, 'validate')
                if validate_entropy < min_validate_entropy:
                    min_validate_entropy = validate_entropy
                if validate_acc > max_validate_acc:
                    max_validate_acc = validate_acc
                    if args.dest:
                        exportSaver.save(sess, args.dest)
                print('Epoch\t%d, Elapsed time\t%.1fs, Accuracy\t%s, Entropy\t%s, Max Accuracy\t%s, Min Entropy\t%s' % (
                    epoch, elapsed_time, validate_acc, validate_entropy, max_validate_acc, min_validate_entropy
                ))
                fd_log.write('{0},{1},{2},{3}\n'.format(
                    epoch, validate_acc, validate_entropy, elapsed_time
                ))
                fd_log.flush()
