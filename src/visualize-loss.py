"""
Usage:
=======
python visualize-loss.py \
    --srcs \
        "..\build\plots\phm2012\lr-phm\log.csv" \
        "..\build\plots\phm2012\rnn-phm\log-1.csv" \
        "..\build\plots\phm2012\rnn-phm\log-2.csv" \
        "..\build\plots\phm2012\rnn-phm\log-3.csv" \
        "..\build\plots\phm2012\rnn-phm\log-4.csv" \
        "..\build\plots\phm2012\rnn-phm\log-5.csv" \
        "..\build\plots\phm2012\rnn-phm\log-6.csv" \
        "..\build\plots\phm2012\rnn-phm\log-7.csv" \
        "..\build\plots\phm2012\rnn-phm\log-8.csv" \
        "..\build\plots\phm2012\rnn-phm\log-9.csv" \
        "..\build\plots\phm2012\rnn-phm\log-10.csv" \
    --labels \
        "linear regression" \
        "rnn-1" \
        "rnn-2 (half batch size)" \
        "rnn-3 (half depth)" \
        "rnn-4 (small timestep size)" \
        "rnn-5 (big timestep size)" \
        "rnn-6 (large timestep size)" \
        "rnn-7 (huge timestep size)" \
        "rnn-8 (double rnn-7 depth)" \
        "rnn-9 (half rnn-7 depth)" \
        "rnn-10 (10x rnn-7 learning rate)" \
    --dest "..\build\plots\phm2012\rnn-phm\loss.png" \
    --x-label "Epochs" \
    --y-label "Train Loss (MSE)" \
    --title "Loss Trend" \
    --sample-size 200
"""
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from utils.utils import get_args, prepare_directory

mpl.rcParams['agg.path.chunksize'] = 10000

args = get_args()

if __name__ == '__main__':
    count = len(args.srcs)
    colors = cm.rainbow(np.linspace(0, 1, count))
    for src, label, color in zip(args.srcs, args.labels, colors):
        df = pd.read_csv(
            src,
            names=['epochs', 'loss', 'elapsed_time']
        )
        sample_size = min(len(df), args.sample_size) if args.sample_size else len(df)
        label = os.path.basename(src).split('.')[0] if label == '' else label
        plt.plot(
            df['epochs'][0:sample_size],
            df['loss'][0:sample_size],
            c=color,
            label=label
        )
    plt.xlabel(args.x_label)
    plt.ylabel(args.y_label)
    plt.title(args.title)
    plt.legend()

    dest_dir = prepare_directory(os.path.dirname(args.dest))
    plt.savefig(
        args.dest,
        dpi=400,
        format='png'
    )
    plt.clf()
