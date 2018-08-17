import os
import glob
import numpy as np
import scipy.stats as st
import matplotlib as mpl
import matplotlib.pyplot as plt

TOTAL = 71758
SEGMENT = 256

if __name__ == '__main__':
    files = glob.glob(os.path.join(__file__, '..', '..', 'build', 'plots', 'journal', 'test-phm-normalized-fft1-classification-256-high-threshold', 'Learning_set-Bearing1_1-acc', '*.eps'))
    first_alarms = np.array(list(map(
        lambda f: round(255 * 0.9) - int(os.path.basename(f).split('.')[0][-4:-1]),
        files
    )))
    scale = TOTAL / SEGMENT
    seconds = first_alarms * scale * 0.39 / 60
    mean = np.mean(seconds)
    ci_95 = st.t.interval(0.95, len(seconds) - 1, loc=mean, scale=st.sem(seconds))

    CI = "%.2f, 95%% CI [%.2f to %.2f]" % (mean, ci_95[0], ci_95[1])
    plt.hist(seconds)
    plt.xlabel('Minutes ahead of real anomaly')
    plt.ylabel('Count')
    plt.title(CI)
    plt.show()
