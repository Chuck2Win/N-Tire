# 넥센 타이어 프로젝트

## Process
- 결측치 제거 및 대체 ➡️ 범주형 변수 처리 ➡️ 모델 학습

## Data 

- 결측치가 많은 경우 제거하고 난 후 data : 15,656개
- raw : 확보 예정 중.

### Feature

총 56개 - 시간(1), 범주형(42), 수치형(13)



|      | Data0 | Data2 | Data3 | Data4 | Data5 | Data9 | Data25 | Data43 | Data44 | Data45 | Data47 | Data49 | Data52 | Data53 |
| ---- | ----- | ----- | ----- | ----- | ----- | ----- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
|      | 15588 | 13    | 16    | 6     | 84    | 8     | 3      | 253    | 14     | 13     | 10     | 14     | 3      | 7      |

- Data0은 제외하고 진행!



- 결측치들이 존재함.
  - 범주형 - 최빈값으로, 수치형 - 중앙값으로 대체
  - 추후 결측치가 많은 데이터가 오게 되면, MICE 혹은 DeepLearning으로 결측치 대체 예정


## Model

- Random Forest
- LightGBM
- Xgboost

:one: non sequential

:two: sequential

![결과](https://github.com/Chuck2Win/N-Tire/blob/main/result/result.PNG)

:arrow_right: sequential로 처리시 R2 score값이 떨어지는 것을 볼 때, 최신의 것을 더 반영한다는 뜻.   
:arrow_right: Data60,Data61는 시간과 무관, Data62,Data63는 시간과 유관


### Important Feature

![결과](https://github.com/Chuck2Win/Nexon-Tire/blob/main/result/6061.png)

![결과](https://github.com/Chuck2Win/Nexon-Tire/blob/main/result/6263.png)



- KFold Validation



--------------



- GridSearch
- Bayesian Optimization



### reference

https://velog.io/@skyepodium/K-Fold-%EA%B5%90%EC%B0%A8%EA%B2%80%EC%A6%9D - kfold

