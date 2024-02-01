import tensorflow as tf
import keras
from itertools import combinations
import math
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Lambda, Add, Activation, Input, Reshape
from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import cv2
import os
import keras.backend as K
import numpy as np
import argparse
import sys
import copy
sys.path.append("../../code")
from ImageNet.Imagenet import ImagenetModel

from tensorflow.python.client import device_lib
device_lib.list_local_devices()


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
hello = tf.constant('Hello Tensorflow')
sess = tf.Session()
attack_left_up_point=(150,150)
# print(sess.run(hello))
# print('hello')

a= tf.constant([[1, 2, 3, 4],
                [1, 2, 3, 4],
                [1, 2, 3, 4],
                [1, 2, 3, 4]])
sess=tf.Session()
b=K.mean(a)

a1=tf.constant([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])
b1=K.mean(a1,axis=-1, keepdims=False)

c=tf.constant([[1],
               [2],
               [3]])

d=tf.constant([[1,2,3]])


print(sess.run(a))
print(a)
print(sess.run(b))
print(b)

print('-------------------')
print(sess.run(a1))
print(a1)
print(sess.run(b1))
print(b1)

print('-------------------')
print(sess.run(c))
print(c)
print('-------------------')
print(sess.run(d))
print(d)


input=(1,2,3)
x1 = tf.random.normal(input)
x2 = tf.random.normal(input)
print(sess.run(x1))
print(sess.run(x2))
print(x1.shape)
print(x2.shape)
y = Add()([x1, x2])
print(sess.run(y))
print(y.shape)