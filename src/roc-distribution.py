"""
Usage:
python roc-distribution.py \
    --dest "..\build\plots\phm2012\roc-distribution\roc-distribution.eps"
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from utils.utils import get_args, prepare_directory

args = get_args()

if __name__ == '__main__':
    FPRs = [
        0.007312274396195139,
        0.008315139127607191,
        0.041436335764388904,
        0.039376906520493644,
        0.019882008640312167,
        0.35177528994595936,
        0.8675152134528732,
    ]
    TPRs = [
        0.22157190635451504,
        0.27046062825750106,
        0.5557120721228342,
        0.5316241724186506,
        0.46992534159740806,
        1.0,
        1.0,
    ]
    text = [
        'Baseline: moving average',
        'RNN Regression',
        '32-segment',
        '64-segment',
        '128-segment',
        '256-segment',
        '512-segment',
    ]
    x_offset = np.repeat(5, 7)
    y_offset = np.repeat(5, 7)

    # 微調
    y_offset[2] = 12
    y_offset[3] = 0
    x_offset[6] = -65
    y_offset[6] = -10

    if args.name == "before experiment":
        FPRs = FPRs[0:2]
        TPRs = TPRs[0:2]
        text = text[0:2]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(FPRs, TPRs, label='models')

    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.tick_params(axis='both', which='minor', labelsize=12)
    plt.title('ROC Space of Anomaly Detection Models', fontsize=20)

    if args.name == "before experiment":
        plt.xlabel('False Alarm Rate', fontsize=20)
        plt.ylabel('True Alarm Rate', fontsize=20)
    else:
        plt.xlabel('False Positive Rate (FPR)', fontsize=20)
        plt.ylabel('True Positive Rate (TPR)', fontsize=20)

    plt.plot([0, 1], [0, 1], '--', label='random')
    plt.legend(fontsize=16, loc='lower right')

    for i, txt in enumerate(text):
        ax.annotate(
            txt,
            (FPRs[i], TPRs[i]),
            va='top',
            xytext=(x_offset[i], y_offset[i]),
            textcoords='offset points',
            fontsize=20
        )

    dest_dir = prepare_directory(os.path.dirname(args.dest))
    plt.savefig(
        args.dest,
        dpi=800,
        format='eps'
    )
