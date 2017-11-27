"""
Usage:
python
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
from utils.utils import log, get_args, prepare_directory
from models.RNN_LSTM import Model

mpl.rcParams['agg.path.chunksize'] = 10000

# parameters
READ_DATA_SIZE = 0

# network parameters
# N_THREADS = 16

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

    df_training = pd.read_csv(args.test_src)

    table = df_training[['x', 'y', 'rulp']].values
    data_size = table.shape[0] - args.step_size + 1

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

    batch_idx = 0
    for sample_from_idx in range(0, data_size):
        batch_idx = batch_idx + 1
        table_sample = table[sample_from_idx: sample_from_idx + args.step_size]

        dataset['x'].append(np.reshape(table_sample[:,0:2], (args.step_size, 2)))
        dataset['y'].append([table_sample[-1, -1]])

        if sample_from_idx % 100000 == 0:
            log('%2.0f%% loaded' % ((float(sample_from_idx) / data_size) * 100))
    dataset_in_order = dataset.copy()
    return dataset

def visualize(model, sess):
    dest_dir = prepare_directory(os.path.join(
        '../build/plots',
        args.scope,
        args.name,
        os.path.basename(args.test_src).rsplit('.', 1)[0]
    ))

    plt.ylim(args.ylim)
    plt.ylabel('Health Index')
    plt.xlabel('Index')
    title = 'Test Health Index'

    x_axis = np.linspace(
        0,
        len(dataset_in_order['y']) - 1,
        num=args.sample_size,
        dtype=int
    )
    ground_truth = np.reshape(
        np.array(dataset_in_order['y'])[x_axis], (args.sample_size)
    )
    ps = model.prediction.eval(
        session=sess,
        feed_dict={
            model.xs: np.array(dataset_in_order['x'])[x_axis],
        }
    )
    predicted = np.reshape(ps, (args.sample_size))

    plt.plot(x_axis, ground_truth, 'g.')
    plt.plot(x_axis, predicted, color='purple', linestyle='--', linewidth=1)

    plt.title(title)
    plt.savefig(
        os.path.join(
            dest_dir,
            'test-health-index-batch_step-{0}.png'.format(args.batch_step)
        ),
        dpi=400,
        format='png'
    )
    plt.clf()

if __name__ == '__main__':
    read_dataset()
    model = Model(
        args.step_size,
        args.input_size,
        args.hidden_size,
        args.output_size,
        args.layer_depth,
        args.batch_size,
        args.dropout_rate,
    )

    # start session
    sess = tf.InteractiveSession(
        # config=tf.ConfigProto(intra_op_parallelism_threads=N_THREADS)
    )

    # prepare model import or export
    importSaver = tf.train.Saver()
    importSaver.restore(sess, args.src)

    visualize(model, sess)
