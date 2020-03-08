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

from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

label_encoder = LabelEncoder()

Y = label_encoder.fit_transform(Y)
Y = to_categorical(Y)

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, shuffle=True)

#############################################################

from keras.layers import Dense, Dropout, BatchNormalization
from keras.models import Sequential

model=Sequential()

model.add(Dense(8, activation='relu',input_shape=(4,)))
model.add(Dense(8, activation='relu'))
model.add(Dense(3,activation='softmax'))

# #3. 훈련
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.fit(X_train,Y_train,batch_size=1,epochs=50,verbose=0, validation_split=0.05)

res= model.evaluate(X_test, Y_test,batch_size=1)
print('Accuracy:', res[1])