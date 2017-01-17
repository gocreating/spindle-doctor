# Data Preprocessing

## Download Dataset

The dataset is too large to upload to github. Please download the dataset from the official site [IEEE PHM 2012 Data Challenge](http://www.femto-st.fr/en/Research-departments/AS2M/Research-groups/PHM/IEEE-PHM-2012-Data-challenge.php). You can put all the downloaded datasets (`Training_set.zip`, `Test_set.zip` and `Full_Test_Set.zip`) into the directory `/phm2012/assets/data/zipped`.(please create directories by yourself if they don't exist)

## Preprocessing Procedure

### 1. Unzip Files

Unzip files into `/phm2012/assets/data/unzipped`.

### 2. Merging Fragmented Files

```
/phm2012/src/preprocess $ python merge.py

```

> #### 注意事項
Temparature 的 data 有些是用分號區隔，不是逗號，所以實作程式時要獨立處理
