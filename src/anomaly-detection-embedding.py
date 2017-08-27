"""
Usage:
python anomaly-detection-embedding.py \
    --scope phm2012 \
    --name ad-phm-embedding \
    --step-size 6 \
    --hidden-size 16 \
    --embedding-size 100 \
    --symbol-size 132 \
    --batch-size 128 \
    --layer-depth 1 \
    --dropout-rate 0.1 \
    --learning-rates \
        1   1000   0.001 \
    --sample-size 128 # must >= batch_size and will be cut to match batch_size
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
from utils.preprocess import unison_shuffled_copies
from models.EncDecEmbedding import Model

mpl.rcParams['agg.path.chunksize'] = 10000

# network parameters
# N_THREADS = 16

Y_LIMIT = [0, 133]

# to be caculated
dataset = {
    'ordered': [],
    'train': [],
    'validate': [],
    'anomalous': [],
}

args = get_args()

def read_dataset():
    global dataset

    file_path = os.path.join(
        '../build/data',
        args.scope, 'labeled/Learning_set-Bearing3_1-acc.csv'
    )
    table = np.genfromtxt(
        file_path,
        delimiter=',',
        skip_header=1,
        usecols=(5,) # level_x
    )
    for sample_from_idx in range(0, table.shape[0] - args.step_size + 1):
        table_sample = table[sample_from_idx: sample_from_idx + args.step_size]
        dataset['ordered'].append(table_sample)
        if sample_from_idx % 100000 == 0:
            log('%2.0f%% loaded' % ((float(sample_from_idx) / (table.shape[0] - args.step_size + 1)) * 100))

    # train, validate split
    dataset_health, dataset['anomalous'] = np.split(dataset['ordered'], [int(.9 * len(dataset['ordered']))])

    # shuffle
    np.random.shuffle(dataset_health)
    dataset['train'], dataset['validate'] = np.split(dataset_health, [int(.9 * len(dataset_health))])
    np.random.shuffle(dataset['train'])

    dataset['ordered'] = np.array(dataset['ordered'])

    # cut last batch
    for name in ['ordered', 'train', 'anomalous', 'validate']:
        d = dataset[name]
        length = len(d)
        r = length % args.batch_size
        dataset[name] = dataset[name][0:-r]

def eval_acc(model, sess, dataset_name):
    accuracy = 0.0
    batch_count, data_size = get_batch_count(dataset[dataset_name], args.batch_size)
    for batch_idx in range(0, batch_count):
        begin_idx = batch_idx * args.batch_size
        end_idx = min(begin_idx + args.batch_size, data_size)
        accuracy = accuracy + model.accuracy.eval(session=sess, feed_dict={
            model.xs: dataset[dataset_name][begin_idx: end_idx],
            model.ys: dataset[dataset_name][begin_idx: end_idx],
            model.feed_previous: True,
        })
    return accuracy / batch_count

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

    accuracy = eval_acc(model, sess, dataset_name)

    title = 'epoch-{0}\n{1} accuracy = {2}'.format(epoch, dataset_name, accuracy)
    plt.title(title)
    plt.savefig(
        os.path.join(dest_dir, 'epoch-{0}-{1}.png'.format(epoch, dataset_name)),
        dpi=400,
        format='png'
    )
    plt.clf()

    return accuracy

if __name__ == '__main__':
    read_dataset()
    model = Model(
        args.step_size,
        args.hidden_size,
        args.embedding_size,
        args.symbol_size,
        args.layer_depth,
        args.batch_size,
        args.dropout_rate
    )

    # start session
    sess = tf.InteractiveSession(
        # config=tf.ConfigProto(intra_op_parallelism_threads=N_THREADS)
    )

    # initize variable
    sess.run(tf.global_variables_initializer())

    filename = args.log or os.path.join(
        prepare_directory(os.path.join(
            '../build/plots', args.scope, args.name)
        ), 'log.csv'
    )
    max_train_acc = 0
    batch_count, data_size = get_batch_count(dataset['train'], args.batch_size)
    with open(filename, 'w') as fd_log:
        start_time = time.time()

        # before training
        train_acc = visualize_dataset(model, sess, 0, 'train')
        anomalous_acc = visualize_dataset(model, sess, 0, 'anomalous')
        validate_acc = eval_acc(model, sess, 'validate')
        print('Epoch\t%d, Batch\t%d, Elapsed time\t%.1fs, Train Acc\t%s, Validate Acc\t%s, Anomalous Acc\t%s, Max Train Acc\t%s' % (
            0, 0, 0, train_acc, validate_acc, anomalous_acc, max_train_acc
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

                    if (batch_idx + 1) % 1000 == 0:
                        elapsed_time = time.time() - start_time
                        print('Epoch\t%d, Batch\t%d, Elapsed time\t%.1fs' % (
                            epoch, batch_idx + 1, elapsed_time
                        ))

                elapsed_time = time.time() - start_time
                train_acc = visualize_dataset(model, sess, epoch, 'train')
                anomalous_acc = visualize_dataset(model, sess, epoch, 'anomalous')
                validate_acc = eval_acc(model, sess, 'validate')
                if train_acc > max_train_acc:
                    max_train_acc = train_acc
                print('Epoch\t%d, Batch\t%d, Elapsed time\t%.1fs, Train Acc\t%s, Validate Acc\t%s, Anomalous Acc\t%s, Max Train Acc\t%s' % (
                    epoch, batch_count, elapsed_time, train_acc, validate_acc, anomalous_acc, max_train_acc
                ))
                fd_log.write('{0},{1},{2},{3},{4}\n'.format(
                    epoch, train_acc, validate_acc, anomalous_acc, elapsed_time
                ))
                fd_log.flush()
