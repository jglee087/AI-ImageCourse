import pandas as pd
from keras.layers import Dense, LSTM, Dropout, BatchNormalization, Input, Activation, Conv2D, MaxPool2D, Flatten
from keras.models import Sequential, Model
from keras.callbacks import TensorBoard, EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler


model=Sequential()
model.add(Conv2D(49, (2,2), padding='same', input_shape=(5,5,1)))
model.add(MaxPool2D(2,2))
model.add(Conv2D(7, (4,4), padding='same'))
model.add(MaxPool2D(2,2))
model.summary()


