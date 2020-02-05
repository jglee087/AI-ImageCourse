# SemiProject 5

## Title: 삼성전자 주가와 KOPSI 200을 사용하여 2월 3일 삼성전자 종가 예측 알고리즘


### 1. 목표

- 삼성전자 종가, 고가, 저가 등과 KOSPI 200 지수의 고가, 저가, 현재가 등을 2018년 5월 4일부터 2020년 1월 31일까지의 데이터를 사용하여 2020년 2월 3일 종가를 예측



### 2. 방법

1. kospi200.csv와 samsung.csv를 불러와서 DataFrame 형태로 저장하고 데이터 전처리를 하고 결과를 Numpy행렬로 저장

2. DNN(Deep Neural Network)모델을 사용하여 주가 예측

3. LSTM(Long-Short Term Memory)모델을 사용하여 주가 예측
4. DNN ensemble 모델을 사용하여 주가 예측
5. LSTM ensemble 모델을 사용하여 주가 예측
6. 20일치 데이터를 시계열 데이터로 묶어 LSTM 모델에 적용



### 3. 결과

- 2월 3일 종가는 57,200원이었고, 모델에 따라 값이 달랐지만 LSTM을 사용한 모델에서 나온 결과값이 종가 값과 가까웠다.