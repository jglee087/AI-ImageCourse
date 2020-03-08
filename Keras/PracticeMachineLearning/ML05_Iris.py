import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

seed=0
np.random.seed(seed)
tf.set_random_seed(seed)

#1. 데이터
try:
    dataset = np.loadtxt("./data/iris.csv",delimiter=',')
except:
    dataset = pd.read_csv("./data/iris.csv",delimiter=',',header=None, index_col=None,encoding='utf-8')
    dataset = dataset.values    
    
X = dataset[:,0:4]
Y = dataset[:,4]

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, shuffle=True)

#############################################################

#2. 모델 (KNeighborsClassifier)
model=KNeighborsClassifier(n_neighbors=10)

#3. 훈련
model.fit(X_train,Y_train)

#4. 평가
y_predict = model.predict(X_test)

#print(X_test,"의 예측 결과:",y_predict)
print("Acc: ", accuracy_score( Y_test, y_predict))

#############################################################

#2. 모델 (Support Vector Classifier)
clf = SVC()

#3. 훈련
clf.fit(X_train,Y_train)

#4. 평가
y_predict = clf.predict(X_test)

#print(X_test,"의 예측 결과:",y_predict)
print("Acc: ", accuracy_score( Y_test, y_predict))

#############################################################

