# Spindle Doctor

A data driven maintainence model for machinery spindle.

## Docs

See [docs](./docs)

## Issues

- PHM data 時間軸精確度不同步
  - 取平均值導致資料失真
  - 如何填補遺失的資料
- 東台 data 資料時間軸出現逆序狀況
  - 導致時間複雜度由 O(n) 變為 O(n2)
- 資料前處理極度費時
  - 5 分鐘的東台 data（約 30 萬筆）需耗費 88 分鐘前處理
  - 10 分鐘的東台 data（約 60 萬筆）需耗費 > 3774 分鐘前處理
  - 考慮使用 Window 概念降低複雜度

## Roadmaps

- [x] 針對不同的 testing dataset，使用不同的 training dataset 訓練 model
- [x] 同時使用溫度值與震動值一起 train model
- [ ] 做投影片
