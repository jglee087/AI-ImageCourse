import tensorflow as tf
import keras

from keras.layers import Dense, LSTM, Dropout, BatchNormalization, Input, Activation, Conv2D, MaxPool2D, Flatten
from keras.models import Sequential, Model

from keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

from keras.datasets import cifar10

img_rows = 32
img_cols = 32

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

input_shape = (img_rows, img_cols, 3)

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
x_test = x_test.reshape(x_test.shape[0],img_rows, img_cols, 3)


x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

x_train = x_train.reshape(-1,32*32*3)
x_test = x_test.reshape(-1,32*32*3)


model=Sequential()

model.add(Dense(512,activation='relu',input_shape=(32*32*3,)))

model.add(Dense(256, kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(256,activation='relu',))
model.add(Dropout(0.5))

model.add(Dense(256,activation='relu'))

model.add(Dense(128, kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(128,activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(64,activation='relu', kernel_initializer='he_normal'))
model.add(Dropout(0.2))

model.add(Dense(64,activation='relu'))

model.add(Dense(32,activation='relu'))

model.add(Dense(32,activation='relu'))

model.add(Dense(10,activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='loss', patience=10)

hist=model.fit(x_train, y_train, batch_size=800, epochs=100, verbose=2, validation_split=0.15, callbacks=[early_stopping] )

import matplotlib.pyplot as plt

print(hist.history.keys())

fig, loss_ax = plt.subplots()
acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
loss_ax.legend(loc='upper left')

acc_ax.plot(hist.history['acc'], 'b', label='train acc')
acc_ax.plot(hist.history['val_acc'], 'g', label='val acc')
acc_ax.set_ylabel('accuracy')
acc_ax.legend(loc='upper right')

plt.show()

loss, acc = model.evaluate(x_test,y_test,batch_size=800)
print('Loss:',loss,'Accuracy:',acc)