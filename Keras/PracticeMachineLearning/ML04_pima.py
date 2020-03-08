import numpy as np
import tensorflow as tf

from keras.layers import Dense, Dropout, BatchNormalization
from keras.models import Sequential

seed=0
np.random.seed(seed)
tf.set_random_seed(seed)

dataset = np.loadtxt("./data/pima-indians-diabetes.csv",delimiter=',')
X = dataset[:,0:8]
Y = dataset[:,8]

model = Sequential()
model.add(Dense(16, activation='relu', input_dim=8))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X,Y, epochs=50, batch_size=4,verbose=4,validation_split=0.1)

print("\n Accuracy: %.4f" % (model.evaluate(X,Y)[1]))