import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

import warnings
warnings.filterwarnings(action='ignore')


wine = pd.read_csv("./data/winequality-white.csv",sep=";", encoding='utf-8')

#1. 데이터

y=wine['quality']
x=wine.drop("quality",axis=1)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8)

#2. 모델 구성
model=RandomForestClassifier()

#3. 훈련
model.fit(x_train,y_train)

res=model.score(x_test,y_test) # Accuracy만 반환
print("Res:",res)

#4. 평가
y_pred = model.predict(x_test)
print("Acc: ", accuracy_score( y_pred, y_test))