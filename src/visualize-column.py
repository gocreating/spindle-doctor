r"""
Usage:
=======
python visualize-column.py \
    --srcs \
        "..\build\data\tongtai\initialized\2017-07-18-168000rpm-working.csv" \
    --columns \
        "datetime" "x" \
    --labels \
        "x_acc" \
    --dest "..\build\plots\tongtai\feature-observation\x.png" \
    --x-label "Time (ms)" \
    --y-label "Acceleration (g)" \
    --title "Acceleration Trend"
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
    columns = np.reshape(
        args.columns,
        (-1, 2)
    )
    for src, label, column, color in zip(args.srcs, args.labels, columns, colors):
        df = pd.read_csv(src)
        sample_size = min(len(df), args.sample_size) if args.sample_size else len(df)
        label = os.path.basename(src).split('.')[0] if label == '' else label
        df['datetime'] = pd.to_datetime(df['datetime']).astype(np.int64) // int(1e6)
        plt.plot(
            df[column[0]][0:sample_size],
            df[column[1]][0:sample_size],
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
