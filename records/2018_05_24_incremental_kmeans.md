# 測試不同的RNN Units

## 訓練模型

```
set CUDA_VISIBLE_DEVICES=1
```

```
python kmeans-anomaly-detection.py \
    --scope phm2012 \
    --name 2018-05-23-phm-normalized-fft1-incremental-k-means-256 \
    --step-size 32 \
    --hidden-size 64 \
    --embedding-size 128 \
    --symbol-size 256 \
    --batch-size 128 \
    --layer-depth 2 \
    --dropout-rate 0.1 \
    --learning-rates \
        1 500 0.001 \
    --sample-size 128 \
    --srcs \
        "../build/data/phm2012/feature-extracted/Learning_set-Bearing1_1-acc.csv" \
        "../build/data/phm2012/feature-extracted/Learning_set-Bearing2_1-acc.csv" \
        "../build/data/phm2012/feature-extracted/Learning_set-Bearing3_1-acc.csv" \
    --columns \
        "normalized_fft1" "anomaly" \
		--src-breakpoint "../build/meta/phm2012/breakpoints-from-feature/breakpoint-256.csv" \
    --dest "../build/models/phm2012/2018-05-23-phm-normalized-fft1-incremental-k-means-256/model"
```
