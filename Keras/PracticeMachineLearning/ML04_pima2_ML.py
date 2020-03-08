import numpy as np
import tensorflow as tf

# LinearSVC, Kneighbors
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from keras.layers import Dense, Dropout, BatchNormalization
from keras.models import Sequential

#1. 데이터
dataset = np.loadtxt("./data/pima-indians-diabetes.csv",delimiter=',')
X = dataset[:,0:8]
Y = dataset[:,8]

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8)


#2. 모델
model=KNeighborsClassifier(n_neighbors=15)

#3. 훈련
model.fit(X_train,Y_train)

#4. 평가
y_predict = model.predict(X_test)

#print(X_test,"의 예측 결과:",y_predict)
print("Acc: ", accuracy_score( Y_test, y_predict))
