# 新增工具機資料

## Adapt
<!--
```
src\phm2012$ python adapt.py
``` -->

## Extract Features

採用z軸資料

```
python extract-feature.py \
    --srcs \
        "../build/data/tongtai-tool-machine/initialized/35354.csv" \
        "../build/data/tongtai-tool-machine/initialized/37447.csv" \
        "../build/data/tongtai-tool-machine/initialized/37832.csv" \
        "../build/data/tongtai-tool-machine/initialized/37833.csv" \
        "../build/data/tongtai-tool-machine/initialized/37834.csv" \
        "../build/data/tongtai-tool-machine/initialized/37835.csv" \
        "../build/data/tongtai-tool-machine/initialized/37836.csv" \
        "../build/data/tongtai-tool-machine/initialized/37837.csv" \
    --dests \
        "../build/data/tongtai-tool-machine/feature-extracted/35354.csv" \
        "../build/data/tongtai-tool-machine/feature-extracted/37447.csv" \
        "../build/data/tongtai-tool-machine/feature-extracted/37832.csv" \
        "../build/data/tongtai-tool-machine/feature-extracted/37833.csv" \
        "../build/data/tongtai-tool-machine/feature-extracted/37834.csv" \
        "../build/data/tongtai-tool-machine/feature-extracted/37835.csv" \
        "../build/data/tongtai-tool-machine/feature-extracted/37836.csv" \
        "../build/data/tongtai-tool-machine/feature-extracted/37837.csv" \
    --batch-size 100
```

## 計算 Break Points

```
python break-point.py \
    --chunk-size 500000 \
    --srcs \
        "..\build\data\tongtai-tool-machine\feature-extracted\35354.csv" \
        "..\build\data\tongtai-tool-machine\feature-extracted\37447.csv" \
        "..\build\data\tongtai-tool-machine\feature-extracted\37832.csv" \
        "..\build\data\tongtai-tool-machine\feature-extracted\37833.csv" \
        "..\build\data\tongtai-tool-machine\feature-extracted\37834.csv" \
        "..\build\data\tongtai-tool-machine\feature-extracted\37835.csv" \
        "..\build\data\tongtai-tool-machine\feature-extracted\37836.csv" \
        "..\build\data\tongtai-tool-machine\feature-extracted\37837.csv" \
    --dest-dir "..\build\meta\tongtai-tool-machine\breakpoints-from-feature" \
    --columns "avg" "fft1" "fft2" "max" "min" "paa" "normalized_fft1" "normalized_paa" \
    --symbol-size 128
python break-point.py \
    --chunk-size 500000 \
    --srcs \
        "..\build\data\tongtai-tool-machine\feature-extracted\35354.csv" \
        "..\build\data\tongtai-tool-machine\feature-extracted\37447.csv" \
        "..\build\data\tongtai-tool-machine\feature-extracted\37832.csv" \
        "..\build\data\tongtai-tool-machine\feature-extracted\37833.csv" \
        "..\build\data\tongtai-tool-machine\feature-extracted\37834.csv" \
        "..\build\data\tongtai-tool-machine\feature-extracted\37835.csv" \
        "..\build\data\tongtai-tool-machine\feature-extracted\37836.csv" \
        "..\build\data\tongtai-tool-machine\feature-extracted\37837.csv" \
    --dest-dir "..\build\meta\tongtai-tool-machine\breakpoints-from-feature" \
    --columns "avg" "fft1" "fft2" "max" "min" "paa" "normalized_fft1" "normalized_paa" \
    --symbol-size 256
```

## 計算 Centroids

<!-- ```
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
    --src-breakpoint "../build/meta/phm2012/breakpoints-from-feature/breakpoint-8.csv" \
    --symbol-size 8
``` -->

## Quantization

```
python extract-feature-level.py \
    --srcs \
        "../build/data/tongtai-tool-machine/feature-extracted/35354.csv" \
        "../build/data/tongtai-tool-machine/feature-extracted/37447.csv" \
        "../build/data/tongtai-tool-machine/feature-extracted/37832.csv" \
        "../build/data/tongtai-tool-machine/feature-extracted/37833.csv" \
        "../build/data/tongtai-tool-machine/feature-extracted/37834.csv" \
        "../build/data/tongtai-tool-machine/feature-extracted/37835.csv" \
        "../build/data/tongtai-tool-machine/feature-extracted/37836.csv" \
        "../build/data/tongtai-tool-machine/feature-extracted/37837.csv" \
    --dests \
        "../build/data/tongtai-tool-machine/feature-128-level-extracted/35354.csv" \
        "../build/data/tongtai-tool-machine/feature-128-level-extracted/37447.csv" \
        "../build/data/tongtai-tool-machine/feature-128-level-extracted/37832.csv" \
        "../build/data/tongtai-tool-machine/feature-128-level-extracted/37833.csv" \
        "../build/data/tongtai-tool-machine/feature-128-level-extracted/37834.csv" \
        "../build/data/tongtai-tool-machine/feature-128-level-extracted/37835.csv" \
        "../build/data/tongtai-tool-machine/feature-128-level-extracted/37836.csv" \
        "../build/data/tongtai-tool-machine/feature-128-level-extracted/37837.csv" \
    --src-breakpoint "../build/meta/tongtai-tool-machine/breakpoints-from-feature/breakpoint-128.csv"
python extract-feature-level.py \
    --srcs \
        "../build/data/tongtai-tool-machine/feature-extracted/35354.csv" \
        "../build/data/tongtai-tool-machine/feature-extracted/37447.csv" \
        "../build/data/tongtai-tool-machine/feature-extracted/37832.csv" \
        "../build/data/tongtai-tool-machine/feature-extracted/37833.csv" \
        "../build/data/tongtai-tool-machine/feature-extracted/37834.csv" \
        "../build/data/tongtai-tool-machine/feature-extracted/37835.csv" \
        "../build/data/tongtai-tool-machine/feature-extracted/37836.csv" \
        "../build/data/tongtai-tool-machine/feature-extracted/37837.csv" \
    --dests \
        "../build/data/tongtai-tool-machine/feature-256-level-extracted/35354.csv" \
        "../build/data/tongtai-tool-machine/feature-256-level-extracted/37447.csv" \
        "../build/data/tongtai-tool-machine/feature-256-level-extracted/37832.csv" \
        "../build/data/tongtai-tool-machine/feature-256-level-extracted/37833.csv" \
        "../build/data/tongtai-tool-machine/feature-256-level-extracted/37834.csv" \
        "../build/data/tongtai-tool-machine/feature-256-level-extracted/37835.csv" \
        "../build/data/tongtai-tool-machine/feature-256-level-extracted/37836.csv" \
        "../build/data/tongtai-tool-machine/feature-256-level-extracted/37837.csv" \
    --src-breakpoint "../build/meta/tongtai-tool-machine/breakpoints-from-feature/breakpoint-256.csv"
```

<!-- ```
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
``` -->

## Model Building

```
python classification-anomaly-detection.py \
    --scope tongtai-tool-machine \
    --name 2018-04-25-normalized-fft1-classification-128 \
    --step-size 32 \
    --hidden-size 64 \
    --embedding-size 128 \
    --symbol-size 128 \
    --batch-size 128 \
    --layer-depth 2 \
    --dropout-rate 0.1 \
    --learning-rates \
        1 500 0.001 \
    --sample-size 128 \
    --srcs \
        "../build/data/tongtai-tool-machine/feature-128-level-extracted/35354.csv" \
        "../build/data/tongtai-tool-machine/feature-128-level-extracted/37447.csv" \
        "../build/data/tongtai-tool-machine/feature-128-level-extracted/37832.csv" \
        "../build/data/tongtai-tool-machine/feature-128-level-extracted/37833.csv" \
        # "../build/data/tongtai-tool-machine/feature-128-level-extracted/37834.csv" \
        # "../build/data/tongtai-tool-machine/feature-128-level-extracted/37835.csv" \
        # "../build/data/tongtai-tool-machine/feature-128-level-extracted/37836.csv" \
        # "../build/data/tongtai-tool-machine/feature-128-level-extracted/37837.csv" \
    --columns \
        "level_normalized_fft1" "anomaly" \
    --src-breakpoint "../build/meta/tongtai-tool-machine/breakpoints-from-feature/breakpoint-128.csv" \
    --dest "../build/models/tongtai-tool-machine/2018-04-25-normalized-fft1-classification-128/model"
python classification-anomaly-detection.py \
    --scope tongtai-tool-machine \
    --name 2018-04-25-normalized-fft1-classification-256 \
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
        "../build/data/tongtai-tool-machine/feature-256-level-extracted/35354.csv" \
        "../build/data/tongtai-tool-machine/feature-256-level-extracted/37447.csv" \
        "../build/data/tongtai-tool-machine/feature-256-level-extracted/37832.csv" \
        "../build/data/tongtai-tool-machine/feature-256-level-extracted/37833.csv" \
        # "../build/data/tongtai-tool-machine/feature-256-level-extracted/37834.csv" \
        # "../build/data/tongtai-tool-machine/feature-256-level-extracted/37835.csv" \
        # "../build/data/tongtai-tool-machine/feature-256-level-extracted/37836.csv" \
        # "../build/data/tongtai-tool-machine/feature-256-level-extracted/37837.csv" \
    --columns \
        "level_normalized_fft1" "anomaly" \
    --src-breakpoint "../build/meta/tongtai-tool-machine/breakpoints-from-feature/breakpoint-256.csv" \
    --dest "../build/models/tongtai-tool-machine/2018-04-25-normalized-fft1-classification-256/model"
```

# Inference

```
python test-classification-anomaly-detection.py \
    --columns "avg" "level_normalized_fft1" "anomaly" \
    --column "level_normalized_fft1" \
    --title "Anomaly Detection by NVM" \
    --threshold 0.0484049507783 \
    --scope tongtai-tool-machine \
    --name test-2018-04-25-normalized-fft1-classification-256 \
    --step-size 32 \
    --hidden-size 64 \
    --embedding-size 128 \
    --symbol-size 256 \
    --batch-size 128 \
    --layer-depth 2 \
    --dropout-rate 0.1 \
    --src-breakpoint "../build/meta/tongtai-tool-machine/breakpoints-from-feature/breakpoint-256.csv" \
    --sample-size 256 \
    --src "../build/models/tongtai-tool-machine/2018-04-25-normalized-fft1-classification-256/model" \
    --test-src "../build/data/tongtai-tool-machine/feature-256-level-extracted/37834.csv" \
    --smooth 15
python test-classification-anomaly-detection.py \
    --columns "avg" "level_normalized_fft1" "anomaly" \
    --column "level_normalized_fft1" \
    --title "Anomaly Detection by NVM" \
    --threshold 0.0484049507783 \
    --scope tongtai-tool-machine \
    --name test-2018-04-15-phm-normalized-fft1-classification-256 \
    --step-size 32 \
    --hidden-size 64 \
    --embedding-size 128 \
    --symbol-size 256 \
    --batch-size 128 \
    --layer-depth 2 \
    --dropout-rate 0.1 \
    --src-breakpoint "../build/meta/tongtai-tool-machine/breakpoints-from-feature/breakpoint-256.csv" \
    --sample-size 256 \
    --src "../build/models/phm2012/2018-04-15-phm-normalized-fft1-classification-256/model" \
    --test-src "../build/data/tongtai-tool-machine/feature-256-level-extracted/37834.csv" \
    --smooth 15
```

# Visualization

```
python visualize-loss.py \
    --srcs \
        "..\build\plots\tongtai-tool-machine\2018-04-25-normalized-fft1-classification-128\log.csv" \
        "..\build\plots\tongtai-tool-machine\2018-04-25-normalized-fft1-classification-128\log.csv" \
        "..\build\plots\tongtai-tool-machine\2018-04-25-normalized-fft1-classification-256\log.csv" \
        "..\build\plots\tongtai-tool-machine\2018-04-25-normalized-fft1-classification-256\log.csv" \
    --labels \
        "128 with anomalous data" \
        "128 with normal data" \
        "256 with anomalous data" \
        "256 with normal data" \
    --names "epochs" "validate_loss" "anomalous_loss" "elapsed_time" \
    --column "elapsed_time" \
    --columns \
        "anomalous_loss" \
        "validate_loss" \
        "anomalous_loss" \
        "validate_loss" \
    --colors "k" "k" "y" "y" \
    --line-styles ":" "_" ":" "_" \
    --markers "x" "x" "o" "o" \
    --markersize 5 5 5 5 \
    --dest "..\build\plots\tongtai-tool-machine\2018-04-25-normalized-fft1\normal-vs-anomalous.eps" \
    --x-label "Training Time (hour)" \
    --y-label "Loss (MSE)" \
    --title "Comparison" \
    --ylim 0.00001 0.1 \
    --sample-size 500
```
