# 利用PHM資料的peak-frequency訓練模型

## 計算 Break Points

```
python break-point.py \
    --chunk-size 500000 \
    --srcs \
        "..\build\data\phm2012\2018-05-29-peak-freq-exrtacted\2018-05-29-Learning_set-Bearing1_1-acc_1280.csv" \
        "..\build\data\phm2012\2018-05-29-peak-freq-exrtacted\2018-05-29-Learning_set-Bearing2_1-acc_1280.csv" \
        "..\build\data\phm2012\2018-05-29-peak-freq-exrtacted\2018-05-29-Learning_set-Bearing3_1-acc_1280.csv" \
    --dest-dir "..\build\meta\phm2012\breakpoints-peak-frequency" \
    --columns "average_acc" "peak_frequency" "normalized_peak_frequency" \
    --symbol-size 256
```

```
python break-point.py \
    --chunk-size 500000 \
    --srcs \
        "..\build\data\phm2012\2018-05-29-peak-freq-exrtacted\Learning_set-Bearing1_1-acc_256.csv" \
        "..\build\data\phm2012\2018-05-29-peak-freq-exrtacted\Learning_set-Bearing2_1-acc_256.csv" \
        "..\build\data\phm2012\2018-05-29-peak-freq-exrtacted\Learning_set-Bearing3_2-acc_256.csv" \
    --dest-dir "..\build\meta\phm2012\breakpoints-peak-frequency" \
    --columns "smooth_acc" "peak_frequency" "normalized_peak_frequency" \
    --symbol-size 256
```

## Quantization

```
python extract-feature-level.py \
    --srcs \
        "..\build\data\phm2012\2018-05-29-peak-freq-exrtacted\2018-05-29-Learning_set-Bearing1_1-acc_1280.csv" \
        "..\build\data\phm2012\2018-05-29-peak-freq-exrtacted\2018-05-29-Learning_set-Bearing2_1-acc_1280.csv" \
        "..\build\data\phm2012\2018-05-29-peak-freq-exrtacted\2018-05-29-Learning_set-Bearing3_1-acc_1280.csv" \
    --dests \
        "../build/data/phm2012/feature-peak-frequency-256-level-extracted/2018-05-29-Learning_set-Bearing1_1-acc_1280.csv" \
        "../build/data/phm2012/feature-peak-frequency-256-level-extracted/2018-05-29-Learning_set-Bearing2_1-acc_1280.csv" \
        "../build/data/phm2012/feature-peak-frequency-256-level-extracted/2018-05-29-Learning_set-Bearing3_1-acc_1280.csv" \
    --src-breakpoint "../build/meta/phm2012/breakpoints-peak-frequency/breakpoint-256.csv"
```

```
python extract-feature-level.py \
    --srcs \
        "..\build\data\phm2012\2018-05-29-peak-freq-exrtacted\Learning_set-Bearing1_1-acc_256.csv" \
        "..\build\data\phm2012\2018-05-29-peak-freq-exrtacted\Learning_set-Bearing2_1-acc_256.csv" \
        "..\build\data\phm2012\2018-05-29-peak-freq-exrtacted\Learning_set-Bearing3_2-acc_256.csv" \
    --dests \
        "../build/data/phm2012/feature-peak-frequency-256-level-extracted/Learning_set-Bearing1_1-acc_256.csv" \
        "../build/data/phm2012/feature-peak-frequency-256-level-extracted/Learning_set-Bearing2_1-acc_256.csv" \
        "../build/data/phm2012/feature-peak-frequency-256-level-extracted/Learning_set-Bearing3_2-acc_256.csv" \
    --src-breakpoint "../build/meta/phm2012/breakpoints-peak-frequency/breakpoint-256.csv"
```

## Model Building

```
python classification-anomaly-detection.py \
    --scope phm2012 \
    --name 2018-05-29-normalized-peak-freq-classification-256 \
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
        "../build/data/phm2012/feature-peak-frequency-256-level-extracted/Learning_set-Bearing1_1-acc_256.csv" \
        "../build/data/phm2012/feature-peak-frequency-256-level-extracted/Learning_set-Bearing2_1-acc_256.csv" \
        "../build/data/phm2012/feature-peak-frequency-256-level-extracted/Learning_set-Bearing3_2-acc_256.csv" \
    --columns \
        "level_normalized_peak_frequency" "anomaly" \
    --src-breakpoint "../build/meta/phm2012/breakpoints-peak-frequency/breakpoint-256.csv" \
    --dest "../build/models/phm2012/2018-05-29-normalized-peak-freq-classification-256/model"
```

# Inference

```
#T = 0.0622063387636371 ~ 0.0607077713725954
python test-classification-anomaly-detection.py \
    --columns "smooth_acc" "level_normalized_peak_frequency" "anomaly" \
    --column "level_normalized_peak_frequency" \
    --title "Anomaly Detection by NVM" \
    --threshold 0.0622063387636371 \
    --scope phm2012 \
    --name test-2018-05-29-normalized-peak-freq-classification-256 \
    --step-size 32 \
    --hidden-size 64 \
    --embedding-size 128 \
    --symbol-size 256 \
    --batch-size 128 \
    --layer-depth 2 \
    --dropout-rate 0.1 \
    --src-breakpoint "../build/meta/phm2012/breakpoints-peak-frequency/breakpoint-256.csv" \
    --sample-size 256 \
    --src "../build/models/phm2012/2018-05-29-normalized-peak-freq-classification-256/model" \
    --test-src "../build/data/phm2012/feature-peak-frequency-256-level-extracted/Learning_set-Bearing1_1-acc_256.csv" \
    --smooth 15 \
    --report-roc
python test-classification-anomaly-detection.py \
    --columns "smooth_acc" "level_normalized_peak_frequency" "anomaly" \
    --column "level_normalized_peak_frequency" \
    --title "Anomaly Detection by NVM" \
    --threshold 0.0622063387636371 \
    --scope phm2012 \
    --name test-2018-05-29-normalized-peak-freq-classification-256 \
    --step-size 32 \
    --hidden-size 64 \
    --embedding-size 128 \
    --symbol-size 256 \
    --batch-size 128 \
    --layer-depth 2 \
    --dropout-rate 0.1 \
    --src-breakpoint "../build/meta/phm2012/breakpoints-peak-frequency/breakpoint-256.csv" \
    --sample-size 256 \
    --src "../build/models/phm2012/2018-05-29-normalized-peak-freq-classification-256/model" \
    --test-src "../build/data/phm2012/feature-peak-frequency-256-level-extracted/Learning_set-Bearing2_1-acc_256.csv" \
    --smooth 15 \
    --report-roc
python test-classification-anomaly-detection.py \
    --columns "smooth_acc" "level_normalized_peak_frequency" "anomaly" \
    --column "level_normalized_peak_frequency" \
    --title "Anomaly Detection by NVM" \
    --threshold 0.0622063387636371 \
    --scope phm2012 \
    --name test-2018-05-29-normalized-peak-freq-classification-256 \
    --step-size 32 \
    --hidden-size 64 \
    --embedding-size 128 \
    --symbol-size 256 \
    --batch-size 128 \
    --layer-depth 2 \
    --dropout-rate 0.1 \
    --src-breakpoint "../build/meta/phm2012/breakpoints-peak-frequency/breakpoint-256.csv" \
    --sample-size 256 \
    --src "../build/models/phm2012/2018-05-29-normalized-peak-freq-classification-256/model" \
    --test-src "../build/data/phm2012/feature-peak-frequency-256-level-extracted/Learning_set-Bearing3_2-acc_256.csv" \
    --smooth 15 \
    --report-roc
```

# Visualization

```
python visualize-loss.py \
    --srcs \
        "..\build\plots\phm2012\2018-05-29-normalized-peak-freq-classification-256\log.csv" \
        "..\build\plots\phm2012\2018-05-29-normalized-peak-freq-classification-256\log.csv" \
    --labels \
        "peak-freq-256 with anomalous data" \
        "peak-freq-256 with normal data" \
    --names "epochs" "validate_loss" "anomalous_loss" "elapsed_time" \
    --column "elapsed_time" \
    --columns \
        "anomalous_loss" \
        "validate_loss" \
    --colors "k" "y" \
    --line-styles ":" "_" \
    --markers "," "D" \
    --markersize 5 5 \
    --dest "..\build\plots\phm2012\2018-05-29-normalized-peak-freq-classification-256\normal-vs-anomalous.eps" \
    --x-label "Training Time (hour)" \
    --y-label "Loss (MSE)" \
    --title "Loss Trend of Model Trained with Peak Frequency" \
    --ylim 0.05 0.175 \
    --sample-size 500 \
    --grid
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
        "1_1" 0 0 "*" "c" \
        "..\build\plots\phm2012\test-2018-05-29-normalized-peak-freq-classification-256\Learning_set-Bearing1_1-acc_256\roc-report-test-2018-05-29-normalized-peak-freq-classification-256(seed=0, smooth=15).csv" \
        \
        "2_1" 0 0 "^" "c" \
        "..\build\plots\phm2012\test-2018-05-29-normalized-peak-freq-classification-256\Learning_set-Bearing2_1-acc_256\roc-report-test-2018-05-29-normalized-peak-freq-classification-256(seed=0, smooth=15).csv" \
        \
        "3_2" 0 0 "x" "c" \
        "..\build\plots\phm2012\test-2018-05-29-normalized-peak-freq-classification-256\Learning_set-Bearing3_2-acc_256\roc-report-test-2018-05-29-normalized-peak-freq-classification-256(seed=0, smooth=15).csv" \
        \
        "Best Proposed" 0 0 "+" "c" \
        "..\build\plots\phm2012\test-phm-normalized-fft1-classification-256-high-threshold\Learning_set-Bearing1_1-acc\roc-report-test-phm-normalized-fft1-classification-256-high-threshold(seed=0, smooth=15).csv" \
    --title "ROC Space of Anomaly Detection Models" \
    --x-label "False Positive Rate (FPR)" \
    --y-label "True Positive Rate (TPR)" \
    --dest "..\build\plots\phm2012\2018-05-29-normalized-peak-freq-classification-256\roc-comparison.eps"
```
