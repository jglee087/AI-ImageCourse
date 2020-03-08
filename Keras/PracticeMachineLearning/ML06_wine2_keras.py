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

from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()
scaler.fit(x)
x=scaler.transform(x)

from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(y)
y = to_categorical(y)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8)


from keras.layers import Dense, Dropout, BatchNormalization
from keras.models import Sequential

model=Sequential()

model.add(Dense(32, activation='relu',input_shape=(11,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(7, activation='softmax'))

# #3. 훈련
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.fit(x_train,y_train,batch_size=8,epochs=50,verbose=0, validation_split=0.2)

res= model.evaluate(x_test, y_test,batch_size=8)
print('Accuracy:', res[1])