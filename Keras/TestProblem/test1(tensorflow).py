#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
import keras
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation

x_data=[1,2,3,4,5,6,7,8,9,10]
y_data=[1,2,3,4,5,6,7,8,9,10]

W = tf.Variable(tf.random_uniform([1], -0.5, 0.5))
b = tf.Variable(tf.random_uniform([1], -0.5, 0.5))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

hypothesis = W * X + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))

a = tf.Variable(0.005)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for step in range(20001):
    sess.run(train, feed_dict={X:x_data, Y:y_data})
    if step % 1000 == 0:
        print('STEP:', step, '   MSE:',sess.run(cost, feed_dict={X:x_data, Y:y_data})) #, sess.run(W), sess.run(b))
        
answer = sess.run(hypothesis, feed_dict={X:[11,12,13]})

print('\nPrediction:',answer)


# In[ ]:




