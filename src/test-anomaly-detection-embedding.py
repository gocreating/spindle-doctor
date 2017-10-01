"""
Usage:
python test-anomaly-detection-embedding.py \
    --scope tongtai \
    --name test-ad-tongtai-embedding \
    --step-size 6 \
    --hidden-size 16 \
    --embedding-size 100 \
    --symbol-size 132 \
    --batch-size 128 \
    --layer-depth 1 \
    --dropout-rate 0.1 \
    --use-column 9 # level_x \
    --src ../build/models/phm2012/ad-phm-embedding/model \
    --test-src ../build/data/tongtai/labeled/2017-08-21-0.5mm-working.csv \
    --batch-step 250
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

# to be caculated
dataset = {
    'ordered': [],
}

args = get_args()

def read_dataset():
    global dataset

    table = np.genfromtxt(
        args.test_src,
        delimiter=',',
        skip_header=1,
        usecols=(args.use_column,)
    )
    for sample_from_idx in range(0, table.shape[0] - args.step_size + 1):
        table_sample = table[sample_from_idx: sample_from_idx + args.step_size]
        dataset['ordered'].append(table_sample)
        if sample_from_idx % 100000 == 0:
            log('%2.0f%% loaded' % ((float(sample_from_idx) / (table.shape[0] - args.step_size + 1)) * 100))

    dataset['ordered'] = np.array(dataset['ordered'])

    # cut last batch
    for name in ['ordered']:
        d = dataset[name]
        length = len(d)
        r = length % args.batch_size
        dataset[name] = dataset[name][0:-r]

def visualize(xs, ys):
    dest_dir = prepare_directory(os.path.join(
        '../build/plots',
        args.scope,
        args.name,
        os.path.basename(args.test_src).rsplit('.', 1)[0]
    ))

    plt.ylim([0, 1])
    plt.ylabel('Accuracy')
    plt.xlabel('Index')
    title = 'Test Accuracy'

    if args.batch_step < 200:
        plt.scatter(xs, ys, color='purple', s=0.1)
    else:
        plt.plot(xs, ys, color='purple', linestyle='--', linewidth=1)

    plt.title(title)
    plt.savefig(
        os.path.join(
            dest_dir,
            'test-accuracy-batch_step-{0}.png'.format(args.batch_step)
        ),
        dpi=400,
        format='png'
    )
    plt.clf()

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

    # prepare model import or export
    importSaver = tf.train.Saver()
    importSaver.restore(sess, args.src)

    filename = args.log or os.path.join(
        prepare_directory(os.path.join(
            '../build/plots', args.scope, args.name)
        ), 'log.csv'
    )
    batch_count, data_size = get_batch_count(dataset['ordered'], args.batch_size)
    plot_xs = []
    plot_ys = []

    start_time = time.time()
    for batch_idx in range(0, batch_count, args.batch_step):
        begin_idx = batch_idx * args.batch_size
        end_idx = min(begin_idx + args.batch_size, data_size)
        ground_truth = dataset['ordered'][begin_idx: end_idx]
        predictions = model.prediction.eval(
            session=sess,
            feed_dict={
                model.xs: ground_truth,
                model.ys: ground_truth,
                model.feed_previous: True,
            }
        )
        errors = np.equal(predictions, ground_truth)
        accuracy = np.mean(errors.astype(int))
        plot_xs.append(end_idx)
        plot_ys.append(accuracy)

        if (batch_idx + 1) % 1000 == 0:
            elapsed_time = time.time() - start_time
            print('Batch\t%d, Elapsed time\t%.1fs' % (
                batch_idx + 1, elapsed_time
            ))

    visualize(plot_xs, plot_ys)
