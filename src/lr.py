"""
Usage:
python lr.py \
    --scope phm2012 \
    --name lr-phm \
    --input-size 2 \
    --batch-size 128 \
    --learning-rates \
        1   1000   0.0001 \
    --sample-size 10000
"""
import os
import time
import math
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from utils.utils import log, get_args, prepare_directory
from utils.preprocess import unison_shuffled_copies
from models.Linear_Regression import Model

mpl.rcParams['agg.path.chunksize'] = 10000

# parameters
READ_DATA_SIZE = 0

# network parameters
# N_THREADS = 16

Y_LIMIT = [0, 1]

# to be caculated
dataset_in_order = {
    'x': [],
    'y': [],
}
dataset = {
    'x': [],
    'y': [],
}
batch_count = 0
data_size = 0

args = get_args()

def read_dataset():
    global data_size
    global batch_count
    global dataset
    global dataset_in_order

    training_file = os.path.join(
        '../build/data',
        args.scope, 'labeled/Learning_set-Bearing3_1-acc.csv'
    )
    df_training = pd.read_csv(
        training_file,
        nrows=(None if READ_DATA_SIZE == 0 else READ_DATA_SIZE)
    )

    table = df_training[['x', 'y', 'rulp']].values
    data_size = table.shape[0]

    # normalization
    x = table[:, 0].flatten()
    y = table[:, 1].flatten()
    mean_x = x.mean()
    mean_y = y.mean()
    std_x = np.std(x)
    std_y = np.std(y)
    table[:, 0] = (x - mean_x) / std_x
    table[:, 1] = (y - mean_y) / std_y

    batch_count = int(math.ceil(float(data_size) / args.batch_size))

    dataset['x'] = table[:, 0:2]
    dataset['y'] = np.reshape(table[:, 2], (-1, 1))

    # shuffle
    dataset_in_order = dataset.copy()
    dataset['x'], dataset['y'] = unison_shuffled_copies(
        np.array(dataset['x']),
        np.array(dataset['y'])
    )
    return dataset

def eval_mse(model, sess):
    r = 0.0
    for batch_idx in range(0, batch_count):
        begin_idx = batch_idx * args.batch_size
        end_idx = min(begin_idx + args.batch_size, data_size)
        r = r + model.error.eval(session=sess, feed_dict={
            model.xs: dataset['x'][begin_idx: end_idx],
            model.ys: dataset['y'][begin_idx: end_idx],
        })
    return r / batch_count

def visualize(model, sess, epoch, train_mse):
    dest_dir = prepare_directory(os.path.join(
        '../build/plots',
        args.scope,
        args.name
    ))

    if args.sample_size:
        x_axis = np.linspace(0, len(dataset_in_order['y']) - 1, num=10000, dtype=int)
        ground_truth = np.reshape(np.array(dataset_in_order['y'])[x_axis], (10000))
        ps = model.prediction.eval(session=sess, feed_dict={
            model.xs: np.array(dataset_in_order['x'])[x_axis],
        })
        predicted = np.reshape(ps, (10000))
    else:
        x_axis = np.arange(data_size)
        ground_truth = np.reshape(dataset_in_order['y'][0:data_size], (data_size))
        ps = np.empty(shape=[0, 1])
        for batch_idx in range(0, batch_count):
            begin_idx = batch_idx * args.batch_size
            end_idx = min(begin_idx + args.batch_size, data_size)

            p = model.prediction.eval(session=sess, feed_dict={
                model.xs: dataset_in_order['x'][begin_idx: end_idx],
            })
            ps = np.concatenate((ps, p), axis=0)
            if (batch_idx + 1) % 5000 == 0:
                log('drawing %d' % (batch_idx + 1))
        predicted = np.reshape(ps, (data_size))

    plt.ylim(Y_LIMIT)
    plt.scatter(x_axis, predicted, color='red', s=1, alpha=0.5, lw=0)
    plt.plot(x_axis, ground_truth, 'g.')
    title = 'epoch-{0}\nmse = {1}'.format(epoch, train_mse)
    plt.title(title)
    plt.savefig(
        os.path.join(dest_dir, 'epoch-{0}.png'.format(epoch)),
        dpi=400,
        format='png'
    )
    plt.clf()

if __name__ == '__main__':
    read_dataset()
    model = Model(
        args.input_size,
        args.batch_size,
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
    min_mse = 999
    with open(filename, 'w') as fd_log:
        start_time = time.time()

        # before training
        train_mse = eval_mse(model, sess)
        visualize(model, sess, 0, train_mse)

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
                            model.xs: dataset['x'][begin_idx: end_idx],
                            model.ys: dataset['y'][begin_idx: end_idx],
                            model.learning_rate: learning_rate,
                        }
                    )

                    if (batch_idx + 1) % 1000 == 0:
                        elapsed_time = time.time() - start_time
                        print('Epoch\t%d, Batch\t%d, Elapsed time\t%.1fs' % (
                            epoch, batch_idx + 1, elapsed_time
                        ))

                elapsed_time = time.time() - start_time
                train_mse = eval_mse(model, sess)
                if train_mse < min_mse:
                    min_mse = train_mse
                print('Epoch\t%d, Batch\t%d, Elapsed time\t%.1fs, MSE\t%s, Min MSE\t%s' % (
                epoch, batch_count, elapsed_time, train_mse, min_mse
                ))
                visualize(model, sess, epoch, train_mse)
                fd_log.write('{0},{1},{2}\n'.format(
                    epoch, train_mse, elapsed_time
                ))
                fd_log.flush()
