# Introduction to PHM 2012 Dataset

The following key points were extracted from the official documentation [IEEE-PHM2012-Challenge-Details](../phm2012/assets/IEEE-PHM2012-Challenge-Details.pdf).

## Feature

The raw dataset contains 2 types of features: `vibration` and `temperature`. Each type of feature is logged with timestamps in different resolution.

| | Column 1 | Column 2 | Column 3 | Column 4 | Column 5 | Column 6 |
| --- | --- | --- | --- | --- | --- | --- |
| **Vibration Signal** | Hour | Minute | Second | μ-second    | Horiz. accel. | vert. accel. |
| **Temparature Signal** | Hour | Minute | Second | 1/10 second | Rtd sensor | - |

## Operating Condition

There are also 3 operating conditions provided.

1. 1800 rpm and 4000 N
2. 1650 rpm and 4200 N
3. 1500 rpm and 5000 N

## Training Instance

3 conditions * 2 instances per condition = 6 run-to-failure dataset instances

## File Organization

```
- Learning_set/
--- Bearing<#Condition>_<#Instance>/
----- acc_00001.csv
----- acc_00002.csv
----- ...
----- acc_02803.csv
----- temp_00001.csv
----- ...
----- temp_00466.csv
- Test_set/
--- Bearing<#Condition>_<#Instance>/
----- ...
- Full_Test_Set/
--- Bearing<#Condition>_<#Instance>/
----- ...
```

## Bearing Life

| Bearing | Recording duration | Actual RUL
| --- | --- | --- |
| Bearing1_1 | 7h47m00s (total life) | - |
| Bearing1_2 | 2h25m00s (total life) | - |
| Bearing2_1 | 2h31m40s (total life) | - |
| Bearing2_2 | 2h12m40s (total life) | - |
| Bearing3_1 | 1h25m40s (total life) | - |
| Bearing3_2 | 4h32m40s (total life) | - |
| - | - | - |
| Bearing1_3 | 5h00m10s | 5730 s |
| Bearing1_4 | 3h09m40s | 339 s  |
| Bearing1_5 | 6h23m30s | 1610 s |
| Bearing1_6 | 6h23m29s | 1460 s |
| Bearing1_7 | 4h10m11s | 7570 s |
| Bearing2_3 | 3h20m10s | 7530 s |
| Bearing2_4 | 1h41m50s | 1390 s |
| Bearing2_5 | 5h33m30s | 3090 s |
| Bearing2_6 | 1h35m10s | 1290 s |
| Bearing2_7 | 0h28m30s | 580 s  |
| Bearing3_3 | 0h58m30s | 820 s  |
