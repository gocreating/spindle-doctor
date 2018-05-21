"""
Usage:
python visualize-norm.py \
    --dest "../build/plots/phm2012/break-points-mapping/break-points-mapping.eps"
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from utils.utils import get_args, prepare_directory

args = get_args()

LOWER_BOUND_X = -2.5
UPPER_BOUND_X = 2.5
PERCENTS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
break_points = list(map(norm.ppf, PERCENTS))
point_count = len(PERCENTS)

xs = np.linspace(LOWER_BOUND_X, UPPER_BOUND_X, 100)
cdf = norm.cdf(xs, 0, 1)

plt.figure(figsize=(10, 8))
plt.plot(xs, cdf, label='Cumulative Distribution Function')
plt.xticks(
    [LOWER_BOUND_X] + break_points[1: -1] + [UPPER_BOUND_X],
    [r'$-\infty$', r'$b_0$', '', '', '', r'$b_i$', '', '', '', r'$b_{k-2}$', r'$\infty$']
)
plt.yticks(
    PERCENTS,
    ['0.0', r'$p_0$', '', '', '', r'$p_i$', '', '', '', r'$p_{k-2}$', '1.0']
)
for i in [1, 5, 9]:
    plt.annotate("",
        xy=(break_points[i], 0), xycoords='data',
        xytext=(LOWER_BOUND_X, PERCENTS[i]), textcoords='data',
        arrowprops=dict(
            arrowstyle='->, head_length=0.7, head_width=0.5',
            connectionstyle='angle, angleA=0, angleB=-90, rad=0',
            linestyle='--',
            color='grey',
            mutation_scale=16
        ),
    )
plt.tick_params(axis='both', which='major', labelsize=18)
plt.ylim([0, 1])
plt.legend(fontsize=16)
plt.title('Break Points Mapping\n', fontsize=20)
plt.xlabel(r'Raw data: $x$', fontsize=20)
plt.ylabel(r'Cumulative probability: $cdf(x)$', fontsize=20)

dest_dir = prepare_directory(os.path.dirname(args.dest))
plt.savefig(
    args.dest,
    dpi=800,
    format='eps'
)
