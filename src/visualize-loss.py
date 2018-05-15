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
    --names \
        "epochs" "train_loss" "validate_loss" "elapsed_time" \
    --column "epochs" \
    --columns \
        "train_loss" \
        "train_loss" \
        "train_loss" \
        "train_loss" \
        "train_loss" \
        "train_loss" \
        "train_loss" \
        "train_loss" \
        "train_loss" \
        "train_loss" \
        "train_loss" \
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
    colors = args.colors or cm.rainbow(np.linspace(0, 1, count))
    # colors = cm.rainbow(np.linspace(0, 1, count))
    for src, label, column, color, line_style, marker, markersize in zip(args.srcs, args.labels, args.columns, colors, args.line_styles, args.markers, args.markersizes):
    # for src, label, column, color in zip(args.srcs, args.labels, args.columns, colors):
        df = pd.read_csv(
            src,
            names=args.names
        )
        sample_size = min(len(df), args.sample_size) if args.sample_size else len(df)
        label = os.path.basename(src).split('.')[0] if label == '' else label
        plt.plot(
            df[args.column][0:sample_size] / 3600,
            df[column][0:sample_size],
            c=color,
            label=label,
            linestyle=line_style.replace('_', '-'),
            linewidth=1,
            marker=marker,
            markevery=20,
            markersize=markersize
        )
    plt.xlabel(args.x_label, fontsize=12)
    plt.ylabel(args.y_label, fontsize=12)
    plt.title(args.title + '\n', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tick_params(axis='both', which='minor', labelsize=12)

    # log y-axis
    if args.log_y_axis:
        dy = 0.00001
        t = np.arange(dy, 1.0, dy)
        plt.ylim(args.ylim)
        plt.semilogy(t, np.exp(-t/5.0))
    if args.grid:
        plt.grid(True)
    # legend_outside = False
    if args.legend_outside:
        lgd = plt.legend(loc='center right', bbox_to_anchor=(args.legend_outside, 0.5), fontsize=10)
    else:
        lgd = plt.legend(fontsize=10)

    dest_dir = prepare_directory(os.path.dirname(args.dest))

    if args.legend_outside:
        plt.savefig(
            args.dest,
            dpi=1200,
            format='eps',
            bbox_extra_artists=(lgd,),
            bbox_inches='tight'
        )
    else:
        plt.savefig(
            args.dest,
            dpi=1200,
            format='eps'
        )
    plt.clf()
