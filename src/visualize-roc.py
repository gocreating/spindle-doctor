"""
Usage:
python visualize-roc.py \
    --srcs \
        "Baseline: moving average" 5 7 \
        "..\build\plots\phm2012\test-2018-04-15-phm-normalized-fft1-moving-average\Learning_set-Bearing1_1-acc\roc-report-test-2018-04-15-phm-normalized-fft1-moving-average(seed=0, smooth=15).csv" \
        "Traditional Regression Model" 5 7 \
        "..\build\plots\phm2012\test-2018-05-15-phm-normalized-fft1-regression\Learning_set-Bearing1_1-acc\roc-report-test-2018-05-15-phm-normalized-fft1-regression(seed=0, smooth=15).csv" \
        "8-segment"  5 -5 \
        "..\build\plots\phm2012\test-2018-04-15-phm-normalized-fft1-classification-8\Learning_set-Bearing1_1-acc\roc-report-test-2018-04-15-phm-normalized-fft1-classification-8(seed=0, smooth=15).csv" \
        "16-segment" -5 27 \
        "..\build\plots\phm2012\test-2018-04-15-phm-normalized-fft1-classification-16\Learning_set-Bearing1_1-acc\roc-report-test-2018-04-15-phm-normalized-fft1-classification-16(seed=0, smooth=15).csv" \
        "32-segment" 5 7 \
        "..\build\plots\phm2012\test-2018-04-15-phm-normalized-fft1-classification-32\Learning_set-Bearing1_1-acc\roc-report-test-2018-04-15-phm-normalized-fft1-classification-32(seed=0, smooth=15).csv" \
        "64-segment" 5 7 \
        "..\build\plots\phm2012\test-2018-04-15-phm-normalized-fft1-classification-64\Learning_set-Bearing1_1-acc\roc-report-test-2018-04-15-phm-normalized-fft1-classification-64(seed=0, smooth=15).csv" \
        "128-segment" 5 7 \
        "..\build\plots\phm2012\test-2018-04-15-phm-normalized-fft1-classification-128\Learning_set-Bearing1_1-acc\roc-report-test-2018-04-15-phm-normalized-fft1-classification-128(seed=0, smooth=15).csv" \
        "256-segment" 5 7 \
        "..\build\plots\phm2012\test-2018-04-15-phm-normalized-fft1-classification-256\Learning_set-Bearing1_1-acc\roc-report-test-2018-04-15-phm-normalized-fft1-classification-256(seed=0, smooth=15).csv" \
    --title "ROC Space of Anomaly Detection Models" \
    --x-label "False Positive Rate (FPR)" \
    --y-label "True Positive Rate (TPR)" \
    --dest "..\build\plots\phm2012\visualized-roc\visualized-roc.eps"
"""
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.utils import get_args, prepare_directory

args = get_args()

if __name__ == '__main__':
    instances = np.reshape(args.srcs, (-1, 6))
    FPRs = []
    TPRs = []

    fig, ax = plt.subplots(figsize=(8, 8))
    for annotate_text, x_offset, y_offset, marker, color, src in instances:
        df_roc_report = pd.read_csv(src, index_col=0)
        FPR = float(df_roc_report['FPR'].values[0])
        TPR = float(df_roc_report['TPR'].values[0])
        if marker == '-':
            FPRs.append(FPR)
            TPRs.append(TPR)
            x_offset = float(x_offset)
            y_offset = float(y_offset)
            ax.annotate(
                annotate_text,
                (FPR, TPR),
                va='top',
                xytext=(x_offset, y_offset),
                textcoords='offset points',
                fontsize=14
            )
        else:
            ax.scatter([FPR], [TPR], s=110, marker=marker, color=color, label=annotate_text)

    if len(FPRs) > 0:
        ax.scatter(FPRs, TPRs, s=110, color='black', label='baseline models')

    # draw perfect model
    ax.scatter([0], [1], s=110, marker='o', color='orange', label='perfect model')

    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tick_params(axis='both', which='minor', labelsize=12)
    plt.title(args.title + '\n', fontsize=20)

    plt.xlabel(args.x_label, fontsize=20)
    plt.ylabel(args.y_label, fontsize=20)

    # draw baseline model
    plt.plot([0, 1], [0, 1], '--', color='grey', label='random model')
    plt.legend(fontsize=14, loc='lower right')

    dest_dir = prepare_directory(os.path.dirname(args.dest))
    plt.savefig(
        args.dest,
        dpi=800,
        format='eps'
    )
