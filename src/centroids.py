r"""
Usage:
=======
python centroids.py \
    --srcs \
        "..\build\data\phm2012\feature-extracted\Learning_set-Bearing1_1-acc.csv" \
        "..\build\data\phm2012\feature-extracted\Learning_set-Bearing1_2-acc.csv" \
        "..\build\data\phm2012\feature-extracted\Learning_set-Bearing2_1-acc.csv" \
        "..\build\data\phm2012\feature-extracted\Learning_set-Bearing2_2-acc.csv" \
        "..\build\data\phm2012\feature-extracted\Learning_set-Bearing3_1-acc.csv" \
        "..\build\data\phm2012\feature-extracted\Learning_set-Bearing3_2-acc.csv" \
    --dest-dir "..\build\meta\phm2012\centroids" \
    --columns "avg" "fft1" "fft2" "max" "min" "paa" "normalized_fft1" "normalized_paa" \
    --src-breakpoint "../build/meta/phm2012/breakpoints-from-feature/breakpoint-256.csv" \
    --symbol-size 256
"""
import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from utils.utils import log, get_args, prepare_directory
from utils.preprocess import breakpoints_to_centroids

def main():
    args = get_args()
    centroids = {}
    df_breakpoints = pd.read_csv(args.src_breakpoint)
    for column in args.columns:
        log('parsing column %s ...' % (column, ))
        data_points = np.array([])
        for src in args.srcs:
            log('parsing %s ...' % (src, ))
            df = pd.read_csv(src, usecols=[column])
            data_points = np.concatenate((data_points, df[column]), axis=0)
        init_centroids = breakpoints_to_centroids(df_breakpoints[column].values)
        kmeans = KMeans(
            init=np.array(init_centroids).reshape((-1, 1)),
            n_clusters=args.symbol_size,
            random_state=0,
            n_jobs=1,
            verbose=0,
            max_iter=500
        ).fit(np.array(data_points).reshape((-1, 1)))
        centroids[column] = np.array(kmeans.cluster_centers_).reshape(-1)

    df_centroids = pd.DataFrame(centroids)
    dest_dir = prepare_directory(args.dest_dir)
    df_centroids.to_csv(
        os.path.join(
            dest_dir,
            'centroid-{0}.csv'.format(args.symbol_size)
        ),
        header=True,
        index=False
    )

main()
