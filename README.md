# 넥센 타이어 프로젝트

## Process
- Feature 선택 ➡️ 결측치 제거 및 대체 ➡️ 범주형 변수 처리 ➡️ 모델 학습

## Data 

- 결측치가 많은 경우 제거하고 난 후 data : 15,656개
- raw : 58,430개 -> 52,063개
1. 종속변수가 결측인 경우 행 제거
2. EDA 진행 후, 이상치가 존재하는 경우 상위 하위 각각 0.001%씩 제거


### Feature

총 56개 - 시간(1), 수치형(42), 범주형(13)



|      |  Data2 | Data3 | Data4 | Data5 | Data9 | Data25 | Data43 | Data44 | Data45 | Data47 | Data49 | Data52 | Data53 |
| ---- |  ----- | ----- | ----- | ----- | ----- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
|      |  13    | 22    | 6     | 160    | 10     | 3      | 442    | 19     | 19     | 15     | 15     | 4      | 11      |


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

https://towardsdatascience.com/custom-transformers-and-ml-data-pipelines-with-python-20ea2a7adb65 - pipeline
