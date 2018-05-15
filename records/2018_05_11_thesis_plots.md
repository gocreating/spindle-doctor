# 繪製碩論用圖片

## RUL Inference Result

```
python label.py \
    --scope phm2012 \
    --thresholds -3.5 3.0 0.05 \
    --src-breakpoint "..\build\meta\phm2012\breakpoints\breakpoint-128.csv"
```

```
python test-rnn.py \
    --scope phm2012 \
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

## RUL Loss Trend

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
    --dest "..\build\plots\phm2012\rnn-phm\hyperparameter-comparison.eps" \
    --x-label "Training Time (Hour)" \
    --y-label "Prediction Error (MSE)" \
    --title "Comparison over Different Hyperparameters" \
    --sample-size 100 \
    --ylim 0.01 0.1
```

## AD Loss Trend

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
    --markersize 5 5 5 5 5 \
    --dest "..\build\plots\phm2012\phm-normalized-fft1\segment-amount-comparison.eps" \
    --x-label "Training Time (Hour)" \
    --y-label "Reconstruct Error (MSE)" \
    --title "Loss Trend of Different Segment Amount" \
    --ylim 0.00001 0.01 \
    --sample-size 300
```

### Linear Regression

| Batch | Learning Rate | min MSE |
| --- | --- | --- |
| 128 | 0.001 | 0.0823917188068 |

### RNN

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
