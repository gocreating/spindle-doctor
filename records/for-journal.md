# Journal用圖表紀錄及重現語法

重複30次測試後計算信心水準

```
# T = 0.0000475455310962373 ~ 0.000107852192698624
python test-classification-anomaly-detection.py \
    --columns "avg" "level_normalized_fft1" "anomaly" \
    --column "level_normalized_fft1" \
    --title "Anomaly Detection by NVM" \
    --threshold 0.000107852192698624 \
    --scope journal \
    --name test-phm-normalized-fft1-classification-256-high-threshold \
    --step-size 32 \
    --hidden-size 64 \
    --embedding-size 128 \
    --symbol-size 256 \
    --batch-size 128 \
    --layer-depth 2 \
    --dropout-rate 0.1 \
    --src-breakpoint "../build/meta/phm2012/breakpoints-from-feature/breakpoint-256.csv" \
    --sample-size 256 \
    --src "../build/models/phm2012/phm-normalized-fft1-classification-256/model" \
    --test-src "../build/data/phm2012/feature-256-level-extracted/Learning_set-Bearing1_1-acc.csv" \
    --smooth 15
    --seed ???
```

```
python confidential-interval.py
```
