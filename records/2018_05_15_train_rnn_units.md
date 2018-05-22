# 測試不同的RNN Units

## 訓練模型

```
set CUDA_VISIBLE_DEVICES=1
```

### 傳統Regression模型

```
python regression-anomaly-detection.py \
    --scope phm2012 \
    --name 2018-05-15-phm-normalized-fft1-regression \
    --step-size 64 \
    --input-size 1 \
    --hidden-size 64 \
    --output-size 1 \
    --batch-size 128 \
    --layer-depth 2 \
    --dropout-rate 0.1 \
    --learning-rates \
        1  500  0.00001 \
    --sample-size 128 \
    --srcs \
        "../build/data/phm2012/feature-extracted/Learning_set-Bearing1_1-acc.csv" \
        "../build/data/phm2012/feature-extracted/Learning_set-Bearing2_1-acc.csv" \
        "../build/data/phm2012/feature-extracted/Learning_set-Bearing3_1-acc.csv" \
    --columns \
        "normalized_fft1" "anomaly" \
    --dest "../build/models/phm2012/2018-05-15-phm-normalized-fft1-regression/model"
```

```
python regression-anomaly-detection.py \
    --scope phm2012 \
    --name 2018-05-15-phm-normalized-fft1-regression-no-shuffle \
    --step-size 64 \
    --input-size 1 \
    --hidden-size 64 \
    --output-size 1 \
    --batch-size 128 \
    --layer-depth 2 \
    --dropout-rate 0.1 \
    --learning-rates \
        1  500  0.00001 \
    --sample-size 128 \
    --no-shuffle \
    --srcs \
        "../build/data/phm2012/feature-extracted/Learning_set-Bearing1_1-acc.csv" \
        "../build/data/phm2012/feature-extracted/Learning_set-Bearing2_1-acc.csv" \
        "../build/data/phm2012/feature-extracted/Learning_set-Bearing3_1-acc.csv" \
    --columns \
        "normalized_fft1" "anomaly" \
    --dest "../build/models/phm2012/2018-05-15-phm-normalized-fft1-regression-no-shuffle-fuck/model"
```

### QUART模型

```
# Basic RNN
python classification-anomaly-detection.py \
    --scope phm2012 \
    --name 2018-05-15-phm-normalized-fft1-classification-256-BASIC-RNN \
    --rnn-unit "BASIC-RNN" \
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
        "../build/data/phm2012/feature-256-level-extracted/Learning_set-Bearing1_1-acc.csv" \
        "../build/data/phm2012/feature-256-level-extracted/Learning_set-Bearing2_1-acc.csv" \
        "../build/data/phm2012/feature-256-level-extracted/Learning_set-Bearing3_1-acc.csv" \
    --columns \
        "level_normalized_fft1" "anomaly" \
    --src-breakpoint "../build/meta/phm2012/breakpoints-from-feature/breakpoint-256.csv" \
    --dest "../build/models/phm2012/2018-05-15-phm-normalized-fft1-classification-256-BASIC-RNN/model"

# GRU
python classification-anomaly-detection.py \
    --scope phm2012 \
    --name 2018-05-15-phm-normalized-fft1-classification-256-GRU \
    --rnn-unit "GRU" \
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
        "../build/data/phm2012/feature-256-level-extracted/Learning_set-Bearing1_1-acc.csv" \
        "../build/data/phm2012/feature-256-level-extracted/Learning_set-Bearing2_1-acc.csv" \
        "../build/data/phm2012/feature-256-level-extracted/Learning_set-Bearing3_1-acc.csv" \
    --columns \
        "level_normalized_fft1" "anomaly" \
    --src-breakpoint "../build/meta/phm2012/breakpoints-from-feature/breakpoint-256.csv" \
    --dest "../build/models/phm2012/2018-05-15-phm-normalized-fft1-classification-256-GRU/model"

# LSTM without shuffling
python classification-anomaly-detection.py \
    --scope phm2012 \
    --name 2018-05-15-phm-normalized-fft1-classification-no-shuffle \
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
        "../build/data/phm2012/feature-256-level-extracted/Learning_set-Bearing1_1-acc.csv" \
        "../build/data/phm2012/feature-256-level-extracted/Learning_set-Bearing2_1-acc.csv" \
        "../build/data/phm2012/feature-256-level-extracted/Learning_set-Bearing3_1-acc.csv" \
    --no-shuffle \
    --columns \
        "level_normalized_fft1" "anomaly" \
    --src-breakpoint "../build/meta/phm2012/breakpoints-from-feature/breakpoint-256.csv" \
    --dest "../build/models/phm2012/2018-05-15-phm-normalized-fft1-classification-no-shuffle/model"
```

## 測試模型

```
# Moving Average
python test-rnn-avg-baseline.py \
    --columns "avg" "normalized_fft1" "anomaly" \
    --column "normalized_fft1" \
    --title "Anomaly Detection by Moving Average" \
    --threshold 0.05 \
    --scope phm2012 \
    --name test-2018-04-15-phm-normalized-fft1-moving-average \
    --step-size 32 \
    --sample-size 256 \
    --report-roc \
    --test-src "../build/data/phm2012/feature-extracted/Learning_set-Bearing1_1-acc.csv" \
    --smooth 15

# Regression 32 step
# T = 0.00850404170644167 ~ 0.00926413834704795
python test-regression-anomaly-detection.py \
    --columns "avg" "normalized_fft1" "anomaly" \
    --column "normalized_fft1" \
    --title "Anomaly Detection by Traditional Model" \
    --threshold 0.00850404170644167 \
    --scope phm2012 \
    --name test-2018-04-15-phm-normalized-fft1-regression \
    --step-size 32 \
    --input-size 1 \
    --hidden-size 64 \
    --output-size 1 \
    --batch-size 128 \
    --layer-depth 2 \
    --dropout-rate 0.1 \
    --sample-size 256 \
    --report-roc \
    --src ../build/models/phm2012/2018-04-15-phm-normalized-fft1-regression/model \
    --test-src ../build/data/phm2012/feature-extracted/Learning_set-Bearing1_1-acc.csv \
    --smooth 15

# Regression 64 step
# T = 0.00276453099152269 ~ 0.00316581708223869
python test-regression-anomaly-detection.py \
    --columns "avg" "normalized_fft1" "anomaly" \
    --column "normalized_fft1" \
    --title "Anomaly Detection by Traditional Model" \
    --threshold 0.00276453099152269 \
    --scope phm2012 \
    --name test-2018-05-15-phm-normalized-fft1-regression \
    --step-size 64 \
    --input-size 1 \
    --hidden-size 64 \
    --output-size 1 \
    --batch-size 128 \
    --layer-depth 2 \
    --dropout-rate 0.1 \
    --sample-size 256 \
    --report-roc \
    --src ../build/models/phm2012/2018-05-15-phm-normalized-fft1-regression/model \
    --test-src ../build/data/phm2012/feature-extracted/Learning_set-Bearing1_1-acc.csv \
    --smooth 15

# 8-segment
# T = 0.0000363608877233636 ~ 0.000135130217333645
python test-classification-anomaly-detection.py \
    --columns "avg" "level_normalized_fft1" "anomaly" \
    --column "level_normalized_fft1" \
    --title "Anomaly Detection by NVM" \
    --threshold 0.0000363608877233636 \
    --scope phm2012 \
    --name test-2018-04-15-phm-normalized-fft1-classification-8 \
    --step-size 32 \
    --hidden-size 64 \
    --embedding-size 128 \
    --symbol-size 8 \
    --batch-size 128 \
    --layer-depth 2 \
    --dropout-rate 0.1 \
    --src-breakpoint "../build/meta/phm2012/breakpoints-from-feature/breakpoint-8.csv" \
    --sample-size 256 \
    --report-roc \
    --src "../build/models/phm2012/2018-04-15-phm-normalized-fft1-classification-8/model" \
    --test-src "../build/data/phm2012/feature-8-level-extracted/Learning_set-Bearing1_1-acc.csv" \
    --smooth 15

# 16-segment
# T = 0.0000542442879058888 ~ 0.000198848241601206
python test-classification-anomaly-detection.py \
    --columns "avg" "level_normalized_fft1" "anomaly" \
    --column "level_normalized_fft1" \
    --title "Anomaly Detection by NVM" \
    --threshold 0.0000542442879058888 \
    --scope phm2012 \
    --name test-2018-04-15-phm-normalized-fft1-classification-16 \
    --step-size 32 \
    --hidden-size 64 \
    --embedding-size 128 \
    --symbol-size 16 \
    --batch-size 128 \
    --layer-depth 2 \
    --dropout-rate 0.1 \
    --src-breakpoint "../build/meta/phm2012/breakpoints-from-feature/breakpoint-16.csv" \
    --sample-size 256 \
    --report-roc \
    --src "../build/models/phm2012/2018-04-15-phm-normalized-fft1-classification-16/model" \
    --test-src "../build/data/phm2012/feature-16-level-extracted/Learning_set-Bearing1_1-acc.csv" \
    --smooth 15

# 32-segment
# T = 0.000172184472214517 ~ 0.000569828932841841
python test-classification-anomaly-detection.py \
    --columns "avg" "level_normalized_fft1" "anomaly" \
    --column "level_normalized_fft1" \
    --title "Anomaly Detection by NVM" \
    --threshold 0.000172184472214517 \
    --scope phm2012 \
    --name test-2018-04-15-phm-normalized-fft1-classification-32 \
    --step-size 32 \
    --hidden-size 64 \
    --embedding-size 128 \
    --symbol-size 32 \
    --batch-size 128 \
    --layer-depth 2 \
    --dropout-rate 0.1 \
    --src-breakpoint "../build/meta/phm2012/breakpoints-from-feature/breakpoint-32.csv" \
    --sample-size 256 \
    --report-roc \
    --src "../build/models/phm2012/2018-04-15-phm-normalized-fft1-classification-32/model" \
    --test-src "../build/data/phm2012/feature-32-level-extracted/Learning_set-Bearing1_1-acc.csv" \
    --smooth 15

# 64-segment
# T = 0.000271929635039073 ~ 0.000789044696995634
python test-classification-anomaly-detection.py \
    --columns "avg" "level_normalized_fft1" "anomaly" \
    --column "level_normalized_fft1" \
    --title "Anomaly Detection by NVM" \
    --threshold 0.000271929635039073 \
    --scope phm2012 \
    --name test-2018-04-15-phm-normalized-fft1-classification-64 \
    --step-size 32 \
    --hidden-size 64 \
    --embedding-size 128 \
    --symbol-size 64 \
    --batch-size 128 \
    --layer-depth 2 \
    --dropout-rate 0.1 \
    --src-breakpoint "../build/meta/phm2012/breakpoints-from-feature/breakpoint-64.csv" \
    --sample-size 256 \
    --report-roc \
    --src "../build/models/phm2012/2018-04-15-phm-normalized-fft1-classification-64/model" \
    --test-src "../build/data/phm2012/feature-64-level-extracted/Learning_set-Bearing1_1-acc.csv" \
    --smooth 15

# 128-segment
# T = 0.000350443765684934 ~ 0.000909460584046938
python test-classification-anomaly-detection.py \
    --columns "avg" "level_normalized_fft1" "anomaly" \
    --column "level_normalized_fft1" \
    --title "Anomaly Detection by NVM" \
    --threshold 0.000350443765684934 \
    --scope phm2012 \
    --name test-2018-04-15-phm-normalized-fft1-classification-128 \
    --step-size 32 \
    --hidden-size 64 \
    --embedding-size 128 \
    --symbol-size 128 \
    --batch-size 128 \
    --layer-depth 2 \
    --dropout-rate 0.1 \
    --src-breakpoint "../build/meta/phm2012/breakpoints-from-feature/breakpoint-128.csv" \
    --sample-size 256 \
    --report-roc \
    --src "../build/models/phm2012/2018-04-15-phm-normalized-fft1-classification-128/model" \
    --test-src "../build/data/phm2012/feature-128-level-extracted/Learning_set-Bearing1_1-acc.csv" \
    --smooth 15

# 256-segment
# T = 0.000403317078173349 ~ 0.00104276275632443
python test-classification-anomaly-detection.py \
    --columns "avg" "level_normalized_fft1" "anomaly" \
    --column "level_normalized_fft1" \
    --title "Anomaly Detection by NVM" \
    --threshold 0.000403317078173349 \
    --scope phm2012 \
    --name test-2018-04-15-phm-normalized-fft1-classification-256 \
    --step-size 32 \
    --hidden-size 64 \
    --embedding-size 128 \
    --symbol-size 256 \
    --batch-size 128 \
    --layer-depth 2 \
    --dropout-rate 0.1 \
    --src-breakpoint "../build/meta/phm2012/breakpoints-from-feature/breakpoint-256.csv" \
    --sample-size 256 \
    --report-roc \
    --src "../build/models/phm2012/2018-04-15-phm-normalized-fft1-classification-256/model" \
    --test-src "../build/data/phm2012/feature-256-level-extracted/Learning_set-Bearing1_1-acc.csv" \
    --smooth 15

# BASIC-RNN
# T = 0.00139878820334 ~ 0.0027274203305956
python test-classification-anomaly-detection.py \
    --columns "avg" "level_normalized_fft1" "anomaly" \
    --column "level_normalized_fft1" \
    --title "Anomaly Detection by QUART (Basic RNN)" \
    --threshold 0.00139878820334 \
    --scope phm2012 \
    --name test-2018-05-15-phm-normalized-fft1-classification-256-BASIC-RNN \
    --step-size 32 \
    --hidden-size 64 \
    --embedding-size 128 \
    --symbol-size 256 \
    --batch-size 128 \
    --layer-depth 2 \
    --dropout-rate 0.1 \
    --src-breakpoint "../build/meta/phm2012/breakpoints-from-feature/breakpoint-256.csv" \
    --sample-size 256 \
    --src "../build/models/phm2012/2018-05-15-phm-normalized-fft1-classification-256-BASIC-RNN/model" \
    --test-src "../build/data/phm2012/feature-256-level-extracted/Learning_set-Bearing1_1-acc.csv" \
    --smooth 15
```

## 繪圖

```
# Shuffle Comparison
python visualize-loss.py \
    --srcs \
        "..\build\plots\phm2012\2018-05-15-phm-normalized-fft1-classification-no-shuffle\log.csv" \
        "..\build\plots\phm2012\2018-05-15-phm-normalized-fft1-classification-no-shuffle\log.csv" \
        "..\build\plots\phm2012\2018-04-15-phm-normalized-fft1-classification-256\log.csv" \
        "..\build\plots\phm2012\2018-04-15-phm-normalized-fft1-classification-256\log.csv" \
    --labels \
        "anomalous data without data shuffling" \
        "normal data without data shuffling" \
        "anomalous data with data shuffling" \
        "normal data with data shuffling" \
    --names "epochs" "validate_loss" "anomalous_loss" "elapsed_time" \
    --column "elapsed_time" \
    --columns \
        "anomalous_loss" \
        "validate_loss" \
        "anomalous_loss" \
        "validate_loss" \
    --colors "k" "k" "y" "y" \
    --line-styles ":" "_" ":" "_" \
    --markers "," "," "D" "D" \
    --markersize 5 5 5 5 \
    --dest "..\build\plots\phm2012\2018-05-15-phm-normalized-fft1-shuffle-comparison\data-shuffle-comparison.eps" \
    --x-label "Training Time (hour)" \
    --y-label "Loss (MSE)" \
    --title "Comparison of with and without Data Shuffling" \
    --grid \
    --log-y-axis \
    --ylim 0.0003 0.02 \
    --sample-size 50

# Cell Comparison
python visualize-loss.py \
    --srcs \
        "..\build\plots\phm2012\2018-05-15-phm-normalized-fft1-classification-256-BASIC-RNN\log.csv" \
        "..\build\plots\phm2012\2018-05-15-phm-normalized-fft1-classification-256-BASIC-RNN\log.csv" \
        "..\build\plots\phm2012\2018-05-15-phm-normalized-fft1-classification-256-GRU\log.csv" \
        "..\build\plots\phm2012\2018-05-15-phm-normalized-fft1-classification-256-GRU\log.csv" \
        "..\build\plots\phm2012\2018-04-15-phm-normalized-fft1-classification-256\log.csv" \
        "..\build\plots\phm2012\2018-04-15-phm-normalized-fft1-classification-256\log.csv" \
    --labels \
        "Basic RNN with anomalous data" \
        "Basic RNN with normal data" \
        "GRU with anomalous data" \
        "GRU with normal data" \
        "LSTM with anomalous data" \
        "LSTM with normal data" \
    --names "epochs" "validate_loss" "anomalous_loss" "elapsed_time" \
    --column "elapsed_time" \
    --columns \
        "anomalous_loss" \
        "validate_loss" \
        "anomalous_loss" \
        "validate_loss" \
        "anomalous_loss" \
        "validate_loss" \
    --colors "k" "k" "y" "y" "c" "c" \
    --line-styles ":" "_" ":" "_" ":" "_" \
    --markers "," "," "D" "D" "x" "x" \
    --markersize 5 5 5 5 5 5 \
    --dest "..\build\plots\phm2012\2018-05-15-phm-normalized-fft1-unit-comparison\rnn-unit-comparison.eps" \
    --x-label "Training Time (hour)" \
    --y-label "Loss (MSE)" \
    --title "Comparison of Different RNN Units" \
    --grid \
    --log-y-axis \
    --ylim 0.0003 0.015 \
    --legend-outside 1.6 \
    --sample-size 300
```

## Parameter Size

regression-2018-04-15(time step 32):  165473
regression-2018-05-15(time step 64):  362113
8-segment:   167432
16-segment:  170000
32-segment:  175136
64-segment:  185408
128-segment: 205952
256-segment: 247040
512-segment: 329216
