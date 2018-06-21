# Custom Activation Function

```
python activation-anomaly-detection.py \
    --scope phm2012 \
    --name 2018-06-21-phm-normalized-fft1-activation-8 \
    --step-size 32 \
    --hidden-size 64 \
    --embedding-size 128 \
    --symbol-size 8 \
    --batch-size 128 \
    --layer-depth 2 \
    --dropout-rate 0.1 \
    --learning-rates \
        1 500 0.001 \
    --sample-size 128 \
    --srcs \
        "../build/data/phm2012/feature-8-level-extracted/Learning_set-Bearing1_1-acc.csv" \
        "../build/data/phm2012/feature-8-level-extracted/Learning_set-Bearing2_1-acc.csv" \
        "../build/data/phm2012/feature-8-level-extracted/Learning_set-Bearing3_1-acc.csv" \
    --columns \
        "level_normalized_fft1" "anomaly" \
    --src-breakpoint "../build/meta/phm2012/breakpoints-from-feature/breakpoint-8.csv" \
    --dest "../build/models/phm2012/2018-06-21-phm-normalized-fft1-activation-8/model"
```

set CUDA_VISIBLE_DEVICES=1

```
python activation-anomaly-detection.py \
    --scope phm2012 \
    --name 2018-06-21-phm-normalized-fft1-activation-16 \
    --step-size 32 \
    --hidden-size 64 \
    --embedding-size 128 \
    --symbol-size 16 \
    --batch-size 128 \
    --layer-depth 2 \
    --dropout-rate 0.1 \
    --learning-rates \
        1 500 0.001 \
    --sample-size 128 \
    --srcs \
        "../build/data/phm2012/feature-16-level-extracted/Learning_set-Bearing1_1-acc.csv" \
        "../build/data/phm2012/feature-16-level-extracted/Learning_set-Bearing2_1-acc.csv" \
        "../build/data/phm2012/feature-16-level-extracted/Learning_set-Bearing3_1-acc.csv" \
    --columns \
        "level_normalized_fft1" "anomaly" \
    --src-breakpoint "../build/meta/phm2012/breakpoints-from-feature/breakpoint-16.csv" \
    --dest "../build/models/phm2012/2018-06-21-phm-normalized-fft1-activation-16/model"
```
