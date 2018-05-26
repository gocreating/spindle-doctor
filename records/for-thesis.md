# 碩論用圖表紀錄及重現語法

## Run-to-failure 振動值

時間長度為 07:47:00，共467分鐘

```
python visualize-column.py \
    --srcs \
        "..\build\data\phm2012\initialized\Learning_set-Bearing1_1-acc.csv" \
    --columns \
        "datetime" "x" \
    --labels \
        "horizontal vibration" \
    --dest "..\build\plots\thesis\run-to-failure-horizontal-vibration-signal.eps" \
    --x-label "Time (ms)" \
    --y-label "Vibration Signal (g)" \
    --title "Run-to-failure Vibration Signal"
```

## 常態分佈

```
python distribution.py \
    --chunk-size 500000 \
    --srcs \
        "..\build\data\phm2012\initialized\Learning_set-Bearing1_1-acc.csv" \
        "..\build\data\phm2012\initialized\Learning_set-Bearing1_2-acc.csv" \
        "..\build\data\phm2012\initialized\Learning_set-Bearing2_1-acc.csv" \
        "..\build\data\phm2012\initialized\Learning_set-Bearing2_2-acc.csv" \
        "..\build\data\phm2012\initialized\Learning_set-Bearing3_1-acc.csv" \
        "..\build\data\phm2012\initialized\Learning_set-Bearing3_2-acc.csv" \
    --dest "..\build\plots\thesis\vibration-distribution.eps" \
    --x-label "Range of Vibration Signal (g)" \
    --y-label "Amount" \
    --title "Vibration Distribution" \
    --thresholds -3.5 3.0 0.05 \
    --column "x"
```

## NVM 斷點計算

```
python visualize-norm.py \
    --dest "../build/plots/thesis/break-points-mapping.eps"
```

## 實驗：RUL模型超參數比較

```
python visualize-loss.py \
    --srcs \
        "..\build\plots\phm2012\rnn-phm\log-1.csv" \
        "..\build\plots\phm2012\rnn-phm\log-2.csv" \
        "..\build\plots\phm2012\rnn-phm\log-3.csv" \
        "..\build\plots\phm2012\rnn-phm\log-4.csv" \
        "..\build\plots\phm2012\rnn-phm\log-5.csv" \
        "..\build\plots\phm2012\rnn-phm\log-6.csv" \
        "..\build\plots\phm2012\rnn-phm\log-7.csv" \
    --labels \
        "case-1: baseline" \
        "case-2: 0.5x batch size" \
        "case-3: 0.5x layer depth" \
        "case-4: 0.5x time step size" \
        "case-5: 2x time step size" \
        "case-6: 4x time step size" \
        "case-7: 8x time step size" \
    --names \
        "epochs" "train_loss" "elapsed_time" \
    --column "elapsed_time" \
    --columns \
        "train_loss" \
        "train_loss" \
        "train_loss" \
        "train_loss" \
        "train_loss" \
        "train_loss" \
        "train_loss" \
    --dest "..\build\plots\thesis\rul-hyperparameter-comparison.eps" \
    --x-label "Training Time (Hour)" \
    --y-label "Prediction Error (MSE)" \
    --title "Comparison over Different Hyperparameters" \
    --sample-size 100 \
    --ylim 0.01 0.075 \
    --auto-line \
    --grid
```

Linear Regression:
| Batch | Learning Rate | min MSE |
| --- | --- | --- |
| 128 | 0.001 | 0.0823917188068 |

RNN:
| # | Cell Activation | Step | Hidden | Batch | Depth | Dropout | Output Activation | Learning Rate | Min Train MSE | sec/epoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | sigmoid | 6 | 64 | 128 | 2 | 0.1 | sigmoid | 0.001 | 0.059108764306 (696-th epoch in 808 epochs) | 95 |
| 2 | sigmoid | 6 | 64 | 64 | 2 | 0.1 | sigmoid | 0.001 | 0.0593994073655 (213-th epoch in 218 epochs) | 172 |
| 3 | sigmoid | 6 | 64 | 128 | 1 | 0.1 | sigmoid | 0.001 | 0.059526798293 (816-th epoch in 914 epochs) | 53 |
| 4 | sigmoid | 4 | 64 | 128 | 2 | 0.1 | sigmoid | 0.001 | 0.0652083003257 (282-th epoch in 292 epochs) | 70 |
| 5 | sigmoid | 16 | 64 | 128 | 2 | 0.1 | sigmoid | 0.001 | 0.0395139667842 (170-th epoch in 170 epochs) | 213 |
| 6 | sigmoid | 32 | 64 | 128 | 2 | 0.1 | sigmoid | 0.001 | 0.0262751034932 (98-th epoch in 100 epochs) | 414 |
| 7 | sigmoid | 64 | 64 | 128 | 2 | 0.1 | sigmoid | 0.001 | 0.01366449 (110-th epoch in 111 epochs) | 778 |
| 8 | sigmoid | 64 | 64 | 128 | 4 | 0.1 | sigmoid | 0.001 | 0.021417787 (16-th epoch in 16 epochs) | 1623 |
| 9 | sigmoid | 64 | 64 | 128 | 1 | 0.1 | sigmoid | 0.001 | 0.014683441 (365-th epoch in 403 epochs) | 400 |
| 10 | sigmoid | 64 | 64 | 128 | 2 | 0.1 | sigmoid | 0.01 | 0.016858411 (135-th epoch in 266 epochs) | 789 |


## 實驗：RUL模型預測結果

```
python label.py \
    --scope phm2012 \
    --thresholds -3.5 3.0 0.05 \
    --src-breakpoint "..\build\meta\phm2012\breakpoints\breakpoint-128.csv"
python test-rnn.py \
    --scope thesis \
    --name test-phm-rul-model-for-phm-data \
    --step-size 64 \
    --input-size 2 \
    --hidden-size 64 \
    --output-size 1 \
    --batch-size 128 \
    --layer-depth 2 \
    --dropout-rate 0.1 \
    --src ../build/models/phm2012/rnn-phm/model \
    --test-src ../build/data/phm2012/labeled/Learning_set-Bearing3_1-acc.csv \
    --sample-size 256 \
    --ylim 0 100 \
    --smooth 15
```

## 實驗：AD Model Parameter Size

regression-2018-04-15(time step 32):  165473
regression-2018-05-15(time step 64):  362113
8-segment:   167432
16-segment:  170000
32-segment:  175136
64-segment:  185408
128-segment: 205952
256-segment: 247040
512-segment: 329216

## 實驗1 - 比較有無Shuffle的差異

247s @3-epoch
93s  @1-epoch
2.7x

```
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
    --colors "k" "k" "c" "c" \
    --line-styles ":" "_" ":" "_" \
    --markers "," "," "D" "D" \
    --markersize 3 3 3 3 \
    --dest "..\build\plots\thesis\with-vs-without-shuffling.eps" \
    --x-label "Training Time (hour)" \
    --y-label "Loss (MSE)" \
    --title "Comparison of with and without Data Shuffling" \
    --grid \
    --log-y-axis \
    --ylim 0.0003 0.02 \
    --sample-size 300
```

## 實驗2：比較有無Quantization的差異

### 實驗2 - NVM

18479s @187-epoch
93s    @1-epoch
198.7x

```
python visualize-loss.py \
    --srcs \
        "..\build\plots\phm2012\2018-05-15-phm-normalized-fft1-regression\log.csv" \
        "..\build\plots\phm2012\2018-05-15-phm-normalized-fft1-regression\log.csv" \
        "..\build\plots\phm2012\2018-04-15-phm-normalized-fft1-classification-256\log.csv" \
        "..\build\plots\phm2012\2018-04-15-phm-normalized-fft1-classification-256\log.csv" \
    --labels \
        "EncDec-AD model with anomalous data" \
        "EncDec-AD model with normal data" \
        "QUART framework with anomalous data" \
        "QUART framework with normal data" \
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
    --markers "o" "o" "v" "v" \
    --markersize 3 3 3 3 \
    --dest "..\build\plots\thesis\with-vs-without-quantization.eps" \
    --x-label "Training Time (hour)" \
    --y-label "Loss (MSE)" \
    --title "Comparison of with and without Quantization" \
    --ylim 0.0001 5 \
    --sample-size 300 \
    --grid \
    --log-y-axis
```

4845s @98-epoch
150s  @1-epoch
32.3x

```
python visualize-loss.py \
    --srcs \
        "..\build\plots\phm2012\phm-normalized-fft1-regression\log.csv" \
        "..\build\plots\phm2012\phm-normalized-fft1-regression\log.csv" \
        "..\build\plots\phm2012\phm-normalized-fft1-classification-128\log.csv" \
        "..\build\plots\phm2012\phm-normalized-fft1-classification-128\log.csv" \
    --labels \
        "No quantization with anomalous data" \
        "No quantization with normal data" \
        "Static quantization using NVM with anomalous data" \
        "Static quantization using NVM with normal data" \
    --names \
        "epochs" "validate_loss" "anomalous_loss" "elapsed_time" \
    --column "elapsed_time" \
    --columns \
        "anomalous_loss" \
        "validate_loss" \
        "anomalous_loss" \
        "validate_loss" \
    --colors "k" "k" "c" "c" \
    --line-styles ":" "_" ":" "_" \
    --markers "," "," "D" "D" \
    --markersize 3 3 3 3 \
    --dest "..\build\plots\thesis\with-vs-without-nvm-static-quantization.eps" \
    --x-label "Training Time (hour)" \
    --y-label "Loss (MSE)" \
    --title "Comparison of with and without NVM Quantization" \
    --ylim 0.0001 1 \
    --sample-size 300 \
    --grid \
    --log-y-axis \
    --legend-location "upper right"
```

### 實驗2 - Linear

```
python visualize-loss.py \
    --srcs \
        "..\build\plots\phm2012\phm-normalized-fft1-classification-equal-256\log.csv" \
        "..\build\plots\phm2012\phm-normalized-fft1-classification-256\log.csv" \
    --labels \
        "Static quantization baseline" \
        "Static quantization using NVM" \
    --names \
        "epochs" "validate_loss" "anomalous_loss" "elapsed_time" \
    --column "elapsed_time" \
    --columns \
        "validate_loss" \
        "validate_loss" \
    --colors "k" "c" \
    --line-styles "_" "_" \
    --markers "," "D" \
    --markersize 3 3 \
    --dest "..\build\plots\thesis\linear-vs-nonlinear.eps" \
    --x-label "Training Time (hour)" \
    --y-label "Loss (MSE)" \
    --title "Loss Trend of Different Static Quantization Mechanisms" \
    --ylim 0.00001 0.001 \
    --sample-size 300 \
    --grid \
    --log-y-axis
```

### 實驗2 - Global K-means (待train)

																FN		FP		FPR									TN		TP		TPR
Global k-means, low threshold		3310	20590	0.31882442204363515	43991	3789	0.5337371460769122
Global k-means, high threshold	3770	18782	0.2908285718709837	45799	3329	0.4689392872235526

```
# T = 0.016389471466757 ~ 0.0239224008033788
python test-classification-anomaly-detection.py \
    --columns "avg" "klevel_normalized_fft1" "anomaly" \
    --column "klevel_normalized_fft1" \
    --title "Anomaly Detection by NVM" \
    --threshold 0.016389471466757 \
    --scope phm2012 \
    --name test-2018-05-23-phm-normalized-fft1-k-means-256-low-threshold \
    --step-size 32 \
    --hidden-size 64 \
    --embedding-size 128 \
    --symbol-size 256 \
    --batch-size 128 \
    --layer-depth 2 \
    --dropout-rate 0.1 \
    --src-centroid "../build/meta/phm2012/centroids/centroid-256.csv" \
    --sample-size 256 \
    --src "../build/models/phm2012/2018-05-23-phm-normalized-fft1-k-means-256/model" \
    --test-src "../build/data/phm2012/feature-256-klevel-extracted/Learning_set-Bearing1_1-acc.csv" \
    --smooth 15 \
    --report-roc
python test-classification-anomaly-detection.py \
    --columns "avg" "klevel_normalized_fft1" "anomaly" \
    --column "klevel_normalized_fft1" \
    --title "Anomaly Detection by Global K-means" \
    --threshold 0.0239224008033788 \
    --scope phm2012 \
    --name test-2018-05-23-phm-normalized-fft1-k-means-256-high-threshold \
    --step-size 32 \
    --hidden-size 64 \
    --embedding-size 128 \
    --symbol-size 256 \
    --batch-size 128 \
    --layer-depth 2 \
    --dropout-rate 0.1 \
    --src-centroid "../build/meta/phm2012/centroids/centroid-256.csv" \
    --sample-size 256 \
    --src "../build/models/phm2012/2018-05-23-phm-normalized-fft1-k-means-256/model" \
    --test-src "../build/data/phm2012/feature-256-klevel-extracted/Learning_set-Bearing1_1-acc.csv" \
    --smooth 15 \
    --report-roc
```

```
python visualize-roc.py \
    --srcs \
        "Moving average" 12 6 "-" "-" \
        "..\build\plots\phm2012\test-2018-04-15-phm-normalized-fft1-moving-average\Learning_set-Bearing1_1-acc\roc-report-test-2018-04-15-phm-normalized-fft1-moving-average(seed=0, smooth=15).csv" \
        \
        "EncDec-AD" 12 6 "-" "-" \
        "..\build\plots\phm2012\test-2018-05-15-phm-normalized-fft1-regression-no-shuffle\Learning_set-Bearing1_1-acc\roc-report-test-2018-05-15-phm-normalized-fft1-regression-no-shuffle(seed=0, smooth=15).csv" \
        \
        "Global K-means" 12 6 "-" "-" \
        "..\build\plots\phm2012\test-2018-05-23-phm-normalized-fft1-k-means-256-low-threshold\Learning_set-Bearing1_1-acc\roc-report-test-2018-05-23-phm-normalized-fft1-k-means-256-low-threshold(seed=0, smooth=15).csv" \
        \
        "Static Quantization (NVM)" 0 0 "*" "c" \
        "..\build\plots\phm2012\test-phm-normalized-fft1-classification-256-high-threshold\Learning_set-Bearing1_1-acc\roc-report-test-phm-normalized-fft1-classification-256-high-threshold(seed=0, smooth=15).csv" \
    --title "ROC Space of Anomaly Detection Models" \
    --x-label "False Positive Rate (FPR)" \
    --y-label "True Positive Rate (TPR)" \
    --dest "..\build\plots\thesis\roc-static-vs-dynamic-comparison.eps"
```

## 實驗3 - 比較不同RNN Units的差異

```
python visualize-loss.py \
    --srcs \
        "..\build\plots\phm2012\2018-05-15-phm-normalized-fft1-classification-256-BASIC-RNN\log.csv" \
        "..\build\plots\phm2012\2018-05-15-phm-normalized-fft1-classification-256-BASIC-RNN\log.csv" \
        "..\build\plots\phm2012\2018-05-15-phm-normalized-fft1-classification-256-GRU\log.csv" \
        "..\build\plots\phm2012\2018-05-15-phm-normalized-fft1-classification-256-GRU\log.csv" \
        "..\build\plots\phm2012\2018-04-15-phm-normalized-fft1-classification-256\log.csv" \
        "..\build\plots\phm2012\2018-04-15-phm-normalized-fft1-classification-256\log.csv" \
    --labels \
        "Vanilla RNN with anomalous data" \
        "Vanilla RNN with normal data" \
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
    --markers "," "," "x" "x" "D" "D" \
    --markersize 3 3 3 3 3 3 \
    --dest "..\build\plots\thesis\vanilla-rnn-vs-gru-vs-lstm.eps" \
    --x-label "Training Time (hour)" \
    --y-label "Loss (MSE)" \
    --title "Comparison of Different RNN Units" \
    --grid \
    --log-y-axis \
    --ylim 0.00045 0.07 \
    --sample-size 300
```

## 實驗4 - 比較有無smoothing的差異

```
python visualize-roc.py \
    --srcs \
        "32-segment (no smoothing)" 0 0 "^" "k" \
        "..\build\plots\phm2012\test-phm-normalized-fft1-classification-32-high-threshold\Learning_set-Bearing1_1-acc\roc-report-test-phm-normalized-fft1-classification-32-high-threshold(seed=0, smooth=1).csv" \
        \
        "64-segment (no smoothing)" 0 0 "d" "k" \
        "..\build\plots\phm2012\test-phm-normalized-fft1-classification-64-high-threshold\Learning_set-Bearing1_1-acc\roc-report-test-phm-normalized-fft1-classification-64-high-threshold(seed=0, smooth=1).csv" \
        \
        "128-segment (no smoothing)" 0 0 "+" "k" \
        "..\build\plots\phm2012\test-phm-normalized-fft1-classification-128-high-threshold\Learning_set-Bearing1_1-acc\roc-report-test-phm-normalized-fft1-classification-128-high-threshold(seed=0, smooth=1).csv" \
        \
        "256-segment (no smoothing)" 0 0 "*" "k" \
        "..\build\plots\phm2012\test-phm-normalized-fft1-classification-256-high-threshold\Learning_set-Bearing1_1-acc\roc-report-test-phm-normalized-fft1-classification-256-high-threshold(seed=0, smooth=1).csv" \
        \
        "512-segment (no smoothing)" 0 0 "x" "k" \
        "..\build\plots\phm2012\test-phm-normalized-fft1-classification-512-high-threshold\Learning_set-Bearing1_1-acc\roc-report-test-phm-normalized-fft1-classification-512-high-threshold(seed=0, smooth=1).csv" \
        \
        "32-segment (smoothed)" 0 0 "^" "c" \
        "..\build\plots\phm2012\test-phm-normalized-fft1-classification-32-high-threshold\Learning_set-Bearing1_1-acc\roc-report-test-phm-normalized-fft1-classification-32-high-threshold(seed=0, smooth=15).csv" \
        \
        "64-segment (smoothed)" 0 0 "d" "c" \
        "..\build\plots\phm2012\test-phm-normalized-fft1-classification-64-high-threshold\Learning_set-Bearing1_1-acc\roc-report-test-phm-normalized-fft1-classification-64-high-threshold(seed=0, smooth=15).csv" \
        \
        "128-segment (smoothed)" 0 0 "+" "c" \
        "..\build\plots\phm2012\test-phm-normalized-fft1-classification-128-high-threshold\Learning_set-Bearing1_1-acc\roc-report-test-phm-normalized-fft1-classification-128-high-threshold(seed=0, smooth=15).csv" \
        \
        "256-segment (smoothed)" 0 0 "*" "c" \
        "..\build\plots\phm2012\test-phm-normalized-fft1-classification-256-high-threshold\Learning_set-Bearing1_1-acc\roc-report-test-phm-normalized-fft1-classification-256-high-threshold(seed=0, smooth=15).csv" \
        \
        "512-segment (smoothed)" 0 0 "x" "c" \
        "..\build\plots\phm2012\test-phm-normalized-fft1-classification-512-high-threshold\Learning_set-Bearing1_1-acc\roc-report-test-phm-normalized-fft1-classification-512-high-threshold(seed=0, smooth=15).csv" \
    --title "ROC Space of Anomaly Detection Models" \
    --x-label "False Positive Rate (FPR)" \
    --y-label "True Positive Rate (TPR)" \
    --dest "..\build\plots\thesis\roc-smoothing-vs-no-smoothing.eps"
```

## 實驗5 - 比較QUART整體改進的效能差異

17880s @185-epoch
150s   @1-epoch
119.2x

```
python visualize-loss.py \
    --srcs \
        "..\build\plots\phm2012\2018-05-15-phm-normalized-fft1-regression-no-shuffle\log.csv" \
        "..\build\plots\phm2012\2018-05-15-phm-normalized-fft1-regression-no-shuffle\log.csv" \
        "..\build\plots\phm2012\2018-04-15-phm-normalized-fft1-classification-256\log.csv" \
        "..\build\plots\phm2012\2018-04-15-phm-normalized-fft1-classification-256\log.csv" \
    --labels \
        "EncDec-AD model with anomalous data" \
        "EncDec-AD model with normal data" \
        "QUART framework with anomalous data" \
        "QUART framework with normal data" \
    --names "epochs" "validate_loss" "anomalous_loss" "elapsed_time" \
    --column "elapsed_time" \
    --columns \
        "anomalous_loss" \
        "validate_loss" \
        "anomalous_loss" \
        "validate_loss" \
    --colors "m" "m" "c" "c" \
    --line-styles ":" "_" ":" "_" \
    --markers "," "," "D" "D" \
    --markersize 3 3 3 3 \
    --dest "..\build\plots\thesis\encdec-ad-vs-quart.eps" \
    --x-label "Training Time (hour)" \
    --y-label "Loss (MSE)" \
    --title "Comparison of EncDec-AD and QUART" \
    --grid \
    --log-y-axis \
    --ylim 0.0003 0.2 \
    --sample-size 300
```

## 實驗6：比較不同segment數量的差異

```
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
    --markersize 3 3 3 3 3 \
    --dest "..\build\plots\thesis\segment-amount-comparison.eps" \
    --x-label "Training Time (Hour)" \
    --y-label "Reconstruct Error (MSE)" \
    --title "Loss Trend of Different Segment Amount" \
    --ylim 0.00001 0.01 \
    --sample-size 300 \
    --grid \
    --log-y-axis
```

												FN		FP		FPR									TN		TP		TPR
32-segment(smooth=1)		2994	2382	0.03688391322525201	62199	4105	0.5782504578109593
32-segment(smooth=15)		2639	2177	0.03370960499218036	62404	4460	0.6282575010564868
64-segment(smooth=1)		3157	7527	0.11655130766014772	57054	3942	0.5552894773911818
64-segment(smooth=15)		2825	8036	0.12443288273640854	56545	4274	0.6020566276940414
128-segment(smooth=1)		3961	4039	0.06254161440671405	60542	3138	0.4420340893083533
128-segment(smooth=15)	3873	4795	0.07424784379306608	59786	3226	0.4544302014368221
256-segment(smooth=1)		0			20879	0.3232994224307459	43702	7099	1.0
256-segment(smooth=15)	0			20797	0.3220296991375172	43784	7099	1.0
512-segment(smooth=1)		0			39237	0.6075625958099131	25344	7099	1.0
512-segment(smooth=15)	0			39988	0.619191403044239		24593	7099	1.0

```
python visualize-roc.py \
    --srcs \
        "Moving average" 12 6 "-" "-" \
        "..\build\plots\phm2012\test-2018-04-15-phm-normalized-fft1-moving-average\Learning_set-Bearing1_1-acc\roc-report-test-2018-04-15-phm-normalized-fft1-moving-average(seed=0, smooth=15).csv" \
        \
        "EncDec-AD" 12 6 "-" "-" \
        "..\build\plots\phm2012\test-2018-05-15-phm-normalized-fft1-regression-no-shuffle\Learning_set-Bearing1_1-acc\roc-report-test-2018-05-15-phm-normalized-fft1-regression-no-shuffle(seed=0, smooth=15).csv" \
        \
        "32-segment" 0 0 "^" "c" \
        "..\build\plots\phm2012\test-phm-normalized-fft1-classification-32-high-threshold\Learning_set-Bearing1_1-acc\roc-report-test-phm-normalized-fft1-classification-32-high-threshold(seed=0, smooth=15).csv" \
        \
        "64-segment" 0 0 "d" "c" \
        "..\build\plots\phm2012\test-phm-normalized-fft1-classification-64-high-threshold\Learning_set-Bearing1_1-acc\roc-report-test-phm-normalized-fft1-classification-64-high-threshold(seed=0, smooth=15).csv" \
        \
        "128-segment" 0 0 "+" "c" \
        "..\build\plots\phm2012\test-phm-normalized-fft1-classification-128-high-threshold\Learning_set-Bearing1_1-acc\roc-report-test-phm-normalized-fft1-classification-128-high-threshold(seed=0, smooth=15).csv" \
        \
        "256-segment" 0 0 "*" "c" \
        "..\build\plots\phm2012\test-phm-normalized-fft1-classification-256-high-threshold\Learning_set-Bearing1_1-acc\roc-report-test-phm-normalized-fft1-classification-256-high-threshold(seed=0, smooth=15).csv" \
        \
        "512-segment" 0 0 "x" "c" \
        "..\build\plots\phm2012\test-phm-normalized-fft1-classification-512-high-threshold\Learning_set-Bearing1_1-acc\roc-report-test-phm-normalized-fft1-classification-512-high-threshold(seed=0, smooth=15).csv" \
    --title "ROC Space of Anomaly Detection Models" \
    --x-label "False Positive Rate (FPR)" \
    --y-label "True Positive Rate (TPR)" \
    --dest "..\build\plots\thesis\roc-segment-amount-comparison.eps"
```

## 實驗7：相容性測試(挑選實驗圖中)

100s @1-epoch
93s  @1-epoch
1.07x

```
python visualize-loss.py \
    --srcs \
        "..\build\plots\phm2012\2018-04-15-phm-normalized-paa-regression-no-shuffle\log.csv" \
        "..\build\plots\phm2012\2018-04-15-phm-normalized-paa-regression-no-shuffle\log.csv" \
        "..\build\plots\phm2012\2018-04-15-phm-normalized-paa-classification-256-3\log.csv" \
        "..\build\plots\phm2012\2018-04-15-phm-normalized-paa-classification-256-3\log.csv" \
    --labels \
        "PAA with anomalous data" \
        "PAA with normal data" \
        "PAA + NVM with anomalous data" \
        "PAA + NVM with normal data" \
    --names "epochs" "validate_loss" "anomalous_loss" "elapsed_time" \
    --column "elapsed_time" \
    --columns \
        "anomalous_loss" \
        "validate_loss" \
        "anomalous_loss" \
        "validate_loss" \
    --colors "k" "k" "c" "c" \
    --line-styles ":" "_" ":" "_" \
    --markers "," "," "D" "D" \
    --markersize 3 3 3 3 \
    --dest "..\build\plots\thesis\paa-vs-paa-plus-nvm.eps" \
    --x-label "Training Time (hour)" \
    --y-label "Loss (MSE)" \
    --title "Loss Trend of Different Data Reduction Approaches" \
    --grid \
    --log-y-axis \
    --ylim 0.0003 0.1 \
    --sample-size 300
```

																FN		FP		FPR										TN		TP		TPR
PAA, low threshold							0			64581	1.0										0			7099	1.0
PAA, high threshold							0			64581	1.0										0			7099	1.0
PAA, no shuffle, low threshold	0			64581	1.0										0			7099	1.0
PAA + NVM, low threshold				0			10841	0.1678667100230718		53740	7099	1.0
PAA + NVM, high threshold				1192	2157	0.033399916384075815	62424	5907	0.8320890266234681
PAA + NVM, low threshold (3)		50		7026	0.10879360802712872		57555	7049	0.9929567544724609
PAA + NVM, high threshold (3)		1864	1221	0.01890648952478283		63360	5235	0.7374278067333427


```
# PAA 64 step, low threshold
# T = 0.00860098651518408 ~ 0.0130754217333603
python test-regression-anomaly-detection.py \
    --columns "avg" "normalized_fft1" "anomaly" \
    --column "normalized_fft1" \
    --title "Anomaly Detection by PAA Model" \
    --threshold 0.00860098651518408 \
    --scope phm2012 \
    --name test-2018-04-15-phm-normalized-paa-regression-low-threshold \
    --step-size 64 \
    --input-size 1 \
    --hidden-size 64 \
    --output-size 1 \
    --batch-size 128 \
    --layer-depth 2 \
    --dropout-rate 0.1 \
    --sample-size 256 \
    --report-roc \
    --src ../build/models/phm2012/2018-04-15-phm-normalized-paa-regression/model \
    --test-src ../build/data/phm2012/feature-extracted/Learning_set-Bearing1_1-acc.csv \
    --smooth 15

# PAA 64 step, high threshold
# T = 0.00860098651518408 ~ 0.0130754217333603
python test-regression-anomaly-detection.py \
    --columns "avg" "normalized_fft1" "anomaly" \
    --column "normalized_fft1" \
    --title "Anomaly Detection by PAA Model" \
    --threshold 0.0130754217333603 \
    --scope phm2012 \
    --name test-2018-04-15-phm-normalized-paa-regression-high-threshold \
    --step-size 64 \
    --input-size 1 \
    --hidden-size 64 \
    --output-size 1 \
    --batch-size 128 \
    --layer-depth 2 \
    --dropout-rate 0.1 \
    --sample-size 256 \
    --report-roc \
    --src ../build/models/phm2012/2018-04-15-phm-normalized-paa-regression/model \
    --test-src ../build/data/phm2012/feature-extracted/Learning_set-Bearing1_1-acc.csv \
    --smooth 15

# PAA 64 step, low threshold, no shuffle
# T = 0.00827427276434671 ~ 0.00987022488178419
python test-regression-anomaly-detection.py \
    --columns "avg" "normalized_fft1" "anomaly" \
    --column "normalized_fft1" \
    --title "Anomaly Detection by PAA Model" \
    --threshold 0.00827427276434671 \
    --scope phm2012 \
    --name test-2018-04-15-phm-normalized-paa-regression-no-shuffle-low-threshold \
    --step-size 64 \
    --input-size 1 \
    --hidden-size 64 \
    --output-size 1 \
    --batch-size 128 \
    --layer-depth 2 \
    --dropout-rate 0.1 \
    --sample-size 256 \
    --report-roc \
    --src ../build/models/phm2012/2018-04-15-phm-normalized-paa-regression-no-shuffle/model \
    --test-src ../build/data/phm2012/feature-extracted/Learning_set-Bearing1_1-acc.csv \
    --smooth 15

# PAA + NVM, low threshold
# T = 0.000999843341186408 ~ 0.00436911697865524
python test-classification-anomaly-detection.py \
    --columns "avg" "level_normalized_fft1" "anomaly" \
    --column "level_normalized_fft1" \
    --title "Anomaly Detection by NVM" \
    --threshold 0.000999843341186408 \
    --scope phm2012 \
    --name test-2018-04-15-phm-normalized-paa-classification-256-low-threshold \
    --step-size 32 \
    --hidden-size 64 \
    --embedding-size 128 \
    --symbol-size 256 \
    --batch-size 128 \
    --layer-depth 2 \
    --dropout-rate 0.1 \
    --src-breakpoint "../build/meta/phm2012/breakpoints-from-feature/breakpoint-256.csv" \
    --sample-size 256 \
    --src "../build/models/phm2012/2018-04-15-phm-normalized-paa-classification-256/model" \
    --test-src "../build/data/phm2012/feature-256-level-extracted/Learning_set-Bearing1_1-acc.csv" \
    --smooth 15 \
    --report-roc

# PAA + NVM, high threshold
# T = 0.000999843341186408 ~ 0.00436911697865524
python test-classification-anomaly-detection.py \
    --columns "avg" "level_normalized_fft1" "anomaly" \
    --column "level_normalized_fft1" \
    --title "Anomaly Detection by NVM" \
    --threshold 0.00436911697865524 \
    --scope phm2012 \
    --name test-2018-04-15-phm-normalized-paa-classification-256-high-threshold \
    --step-size 32 \
    --hidden-size 64 \
    --embedding-size 128 \
    --symbol-size 256 \
    --batch-size 128 \
    --layer-depth 2 \
    --dropout-rate 0.1 \
    --src-breakpoint "../build/meta/phm2012/breakpoints-from-feature/breakpoint-256.csv" \
    --sample-size 256 \
    --src "../build/models/phm2012/2018-04-15-phm-normalized-paa-classification-256/model" \
    --test-src "../build/data/phm2012/feature-256-level-extracted/Learning_set-Bearing1_1-acc.csv" \
    --smooth 15 \
    --report-roc

# PAA + NVM - 3, low threshold
# T = 0.0024559026141156 ~ 0.00771249597015517
python test-classification-anomaly-detection.py \
    --columns "avg" "level_normalized_fft1" "anomaly" \
    --column "level_normalized_fft1" \
    --title "Anomaly Detection by NVM" \
    --threshold 0.0024559026141156 \
    --scope phm2012 \
    --name test-2018-04-15-phm-normalized-paa-classification-256-3-low-threshold \
    --step-size 32 \
    --hidden-size 64 \
    --embedding-size 128 \
    --symbol-size 256 \
    --batch-size 128 \
    --layer-depth 2 \
    --dropout-rate 0.1 \
    --src-breakpoint "../build/meta/phm2012/breakpoints-from-feature/breakpoint-256.csv" \
    --sample-size 256 \
    --src "../build/models/phm2012/2018-04-15-phm-normalized-paa-classification-256-3/model" \
    --test-src "../build/data/phm2012/feature-256-level-extracted/Learning_set-Bearing1_1-acc.csv" \
    --smooth 15 \
    --report-roc

# PAA + NVM - 3, high threshold
# T = 0.0024559026141156 ~ 0.00771249597015517
python test-classification-anomaly-detection.py \
    --columns "avg" "level_normalized_fft1" "anomaly" \
    --column "level_normalized_fft1" \
    --title "Anomaly Detection by NVM" \
    --threshold 0.00771249597015517 \
    --scope phm2012 \
    --name test-2018-04-15-phm-normalized-paa-classification-256-3-high-threshold \
    --step-size 32 \
    --hidden-size 64 \
    --embedding-size 128 \
    --symbol-size 256 \
    --batch-size 128 \
    --layer-depth 2 \
    --dropout-rate 0.1 \
    --src-breakpoint "../build/meta/phm2012/breakpoints-from-feature/breakpoint-256.csv" \
    --sample-size 256 \
    --src "../build/models/phm2012/2018-04-15-phm-normalized-paa-classification-256-3/model" \
    --test-src "../build/data/phm2012/feature-256-level-extracted/Learning_set-Bearing1_1-acc.csv" \
    --smooth 15 \
    --report-roc
```

```
python visualize-roc.py \
    --srcs \
        "Moving average" 12 6 "-" "-" \
        "..\build\plots\phm2012\test-2018-04-15-phm-normalized-fft1-moving-average\Learning_set-Bearing1_1-acc\roc-report-test-2018-04-15-phm-normalized-fft1-moving-average(seed=0, smooth=15).csv" \
        \
        "EncDec-AD" 12 6 "-" "-" \
        "..\build\plots\phm2012\test-2018-05-15-phm-normalized-fft1-regression-no-shuffle\Learning_set-Bearing1_1-acc\roc-report-test-2018-05-15-phm-normalized-fft1-regression-no-shuffle(seed=0, smooth=15).csv" \
        \
        "NVM" 0 0 "*" "c" \
        "..\build\plots\phm2012\test-phm-normalized-fft1-classification-256-high-threshold\Learning_set-Bearing1_1-acc\roc-report-test-phm-normalized-fft1-classification-256-high-threshold(seed=0, smooth=15).csv" \
        \
        "PAA" 0 0 "^" "c" \
        "..\build\plots\phm2012\test-2018-04-15-phm-normalized-paa-regression-low-threshold\Learning_set-Bearing1_1-acc\roc-report-test-2018-04-15-phm-normalized-paa-regression-low-threshold(seed=0, smooth=15).csv" \
        \
        "PAA+NVM" 0 0 "x" "c" \
        "..\build\plots\phm2012\test-2018-04-15-phm-normalized-paa-classification-256-3-low-threshold\Learning_set-Bearing1_1-acc\roc-report-test-2018-04-15-phm-normalized-paa-classification-256-3-low-threshold(seed=0, smooth=15).csv" \
    --title "ROC Space of Anomaly Detection Models" \
    --x-label "False Positive Rate (FPR)" \
    --y-label "True Positive Rate (TPR)" \
    --dest "..\build\plots\thesis\roc-compatibility-comparison.eps"
```
