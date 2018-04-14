# 整理 Paper 圖片

``` bash
python visualize-column.py \
    --srcs \
        "..\build\data\phm2012\initialized\Learning_set-Bearing1_1-acc.csv" \
    --columns \
        "datetime" "x" \
    --labels \
        "horizontal vibration" \
    --dest "..\build\plots\phm2012\feature-observation\Learning_set-Bearing1_1-acc\vibration-signal.eps" \
    --x-label "Time" \
    --y-label "Vibration Signal" \
    --title "Run-to-failure Vibration Signal"
```

``` bash
# normal-vs-anomalous
python visualize-loss.py \
    --srcs \
        "..\build\plots\phm2012\phm-normalized-fft1-regression\log.csv" \
        "..\build\plots\phm2012\phm-normalized-fft1-regression\log.csv" \
        "..\build\plots\phm2012\phm-normalized-fft1-classification-equal-128\log.csv" \
        "..\build\plots\phm2012\phm-normalized-fft1-classification-128\log.csv" \
    --labels \
        "RNN regression (anomalous)" \
        "RNN regression (normal)" \
        "NVM (anomalous)" \
        "NVM (normal)" \
    --names \
        "epochs" "validate_loss" "anomalous_loss" "elapsed_time" \
    --column "elapsed_time" \
    --columns \
        "anomalous_loss" \
        "validate_loss" \
        "anomalous_loss" \
        "validate_loss" \
    --colors "m" "m" "c" "c" \
    --line-styles ":" "_" ":" "_" \
    --markers "." "." "." "." \
    --markersize 0 0 0 0 \
    --dest "..\build\plots\phm2012\phm-normalized-fft1\normal-vs-anomalous.eps" \
    --x-label "Training Time (s)" \
    --y-label "Loss (MSE)" \
    --title "Comparison of Traditional Approach And Proposed Mechanism" \
    --ylim 0.0001 1 \
    --sample-size 300

# equal-vs-unequal
python visualize-loss.py \
    --srcs \
        "..\build\plots\phm2012\phm-normalized-fft1-classification-equal-128\log.csv" \
        "..\build\plots\phm2012\phm-normalized-fft1-classification-128\log.csv" \
    --labels \
        "baseline (linear mapping)" \
        "NVM (nonlinear mapping)" \
    --names \
        "epochs" "validate_loss" "anomalous_loss" "elapsed_time" \
    --column "elapsed_time" \
    --columns \
        "validate_loss" \
        "validate_loss" \
    --colors "b" "c" \
    --line-styles "_" "_" \
    --markers "." "." \
    --markersize 0 0 \
    --dest "..\build\plots\phm2012\phm-normalized-fft1\equal-vs-unequal.eps" \
    --x-label "Training Time (s)" \
    --y-label "Loss (MSE)" \
    --title "Loss Trend of Different Segment Approach" \
    --ylim 0.0001 0.01 \
    --sample-size 300

# segment-amount-comparison
python visualize-loss.py \
    --srcs \
        "..\build\plots\phm2012\phm-normalized-fft1-classification-32\log.csv" \
        "..\build\plots\phm2012\phm-normalized-fft1-classification-64\log.csv" \
        "..\build\plots\phm2012\phm-normalized-fft1-classification-128\log.csv" \
        "..\build\plots\phm2012\phm-normalized-fft1-classification-256\log.csv" \
        "..\build\plots\phm2012\phm-normalized-fft1-classification-512\log.csv" \
    --labels \
        "32-segment" \
        "64-segment" \
        "128-segment" \
        "256-segment" \
        "512-segment" \
    --names \
        "epochs" "validate_loss" "anomalous_loss" "elapsed_time" \
    --column "elapsed_time" \
    --columns \
        "validate_loss" \
        "validate_loss" \
        "validate_loss" \
        "validate_loss" \
        "validate_loss" \
    --colors "k" "y" "c" "tab:orange" "tab:pink" \
    --line-styles "_" "_" "_" "_" "_" \
    --markers "," "D" "x" "o" "v" \
    --markersize 5 5 5 5 5 \
    --dest "..\build\plots\phm2012\phm-normalized-fft1\segment-amount-comparison.eps" \
    --x-label "Time(s)" \
    --y-label "Loss (MSE)" \
    --title "Loss Trend of Different Segment Amount" \
    --ylim 0.00001 0.01 \
    --sample-size 300
```

``` bash
python roc-distribution.py \
    --name "after experiment" \
    --dest "..\build\plots\phm2012\roc-distribution\roc-distribution-after-experiment.eps"
```

``` bash
# RNN Regression
python test-regression-anomaly-detection.py \
    --columns "avg" "normalized_fft1" \
    --column "normalized_fft1" \
    --title "Anomaly Detection by RNN Regression" \
    --threshold 0.00813998495383809 \
    --scope phm2012 \
    --name test-phm-normalized-fft1-regression \
    --step-size 32 \
    --input-size 1 \
    --hidden-size 64 \
    --output-size 1 \
    --batch-size 128 \
    --layer-depth 2 \
    --dropout-rate 0.1 \
    --sample-size 256 \
    --src ../build/models/phm2012/phm-normalized-fft1-regression/model \
    --test-src ../build/data/phm2012/feature-extracted/Learning_set-Bearing1_1-acc.csv

# unequal-256-level
python test-classification-anomaly-detection.py \
    --columns "avg" "level_normalized_fft1" "anomaly" \
    --column "level_normalized_fft1" \
    --title "Anomaly Detection by NVM" \
    --threshold 0.000107852192698624 \
    --scope phm2012 \
    --name test-phm-normalized-fft1-classification-256 \
    --step-size 32 \
    --hidden-size 64 \
    --embedding-size 128 \
    --symbol-size 256 \
    --batch-size 128 \
    --layer-depth 2 \
    --dropout-rate 0.1 \
    --src-breakpoint "../build/meta/phm2012/breakpoints/breakpoint-256.csv" \
    --sample-size 256 \
    --src "../build/models/phm2012/phm-normalized-fft1-classification-256/model" \
    --test-src "../build/data/phm2012/feature-256-level-extracted/Learning_set-Bearing1_1-acc.csv" \
    --seed 9999
```
