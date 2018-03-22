"""
Usage:
python extract-kmeans-level.py \
    --srcs \
        "../build/data/phm2012/feature-extracted/Learning_set-Bearing1_1-acc.csv" \
        "../build/data/phm2012/feature-extracted/Learning_set-Bearing1_2-acc.csv" \
        "../build/data/phm2012/feature-extracted/Learning_set-Bearing2_1-acc.csv" \
        "../build/data/phm2012/feature-extracted/Learning_set-Bearing2_2-acc.csv" \
        "../build/data/phm2012/feature-extracted/Learning_set-Bearing3_1-acc.csv" \
        "../build/data/phm2012/feature-extracted/Learning_set-Bearing3_2-acc.csv" \
    --dests \
        "../build/data/phm2012/feature-256-klevel-extracted/Learning_set-Bearing1_1-acc.csv" \
        "../build/data/phm2012/feature-256-klevel-extracted/Learning_set-Bearing1_2-acc.csv" \
        "../build/data/phm2012/feature-256-klevel-extracted/Learning_set-Bearing2_1-acc.csv" \
        "../build/data/phm2012/feature-256-klevel-extracted/Learning_set-Bearing2_2-acc.csv" \
        "../build/data/phm2012/feature-256-klevel-extracted/Learning_set-Bearing3_1-acc.csv" \
        "../build/data/phm2012/feature-256-klevel-extracted/Learning_set-Bearing3_2-acc.csv" \
    --columns "avg" "fft1" "fft2" "max" "min" "paa" "normalized_fft1" "normalized_paa" \
    --src-centroid "..\build\meta\phm2012\centroids\centroid-256.csv"
"""
import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from utils.utils import log, prepare_directory, get_args
from utils.input import get_batch

def main():
    args = get_args()
    df_centroids = pd.read_csv(args.src_centroid)

    # construct kmeans models
    kmeans_models = {}
    for column in args.columns:
        init_centroids = np.array(df_centroids[column]).reshape((-1, 1))
        kmeans_models[column] = KMeans(
            init=init_centroids,
            n_clusters=len(df_centroids),
            random_state=0,
            n_jobs=1,
            verbose=0,
            max_iter=1
        ).fit(init_centroids)

    for src, dest in zip(args.srcs, args.dests):
        log('parsing %s ...' % (src, ))
        prepare_directory(os.path.dirname(dest))
        df_chunks = pd.read_csv(
            src,
            chunksize=args.chunk_size
        )
        for chunk_idx, df_chunk in enumerate(df_chunks):
            if chunk_idx % 1 == 0:
                print(chunk_idx)
            for column in args.columns:
                assignments = kmeans_models[column].predict(np.array(df_chunk[column]).reshape((-1, 1)))
                df_chunk['klevel_' + column] = assignments

            header = chunk_idx is 0
            mode = 'a' if chunk_idx > 0 else 'w'
            df_chunk.to_csv(
                os.path.join(dest),
                mode=mode,
                header=header,
                index=False
            )

main()
