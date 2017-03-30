#!/usr/bin/python
## Author: sumanth
## Date: March 23, 2017
# model to train the data

# Python 2/3 compatibility
from __future__ import print_function

from keras.layers import Dense, Flatten, Lambda, Activation, MaxPooling2D
from keras.layers.convolutional import Convolution2D
from keras.models import Sequential
import json
from keras.layers import ELU

mode_path='./model_weights/'

# model is based on NVIDIA's "End to End Learning for Self-Driving Cars" paper
model = Sequential()

model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(66, 220, 3)))

# starts with five convolutional and maxpooling layers
model.add(Convolution2D(24, 5, 5, activation='relu', subsample = (2, 2), border_mode = 'same', init="glorot_uniform", bias = True))

model.add(Convolution2D(36, 5, 5, activation='relu', subsample = (2, 2), border_mode = 'same', init="glorot_uniform", bias = True))

model.add(Convolution2D(48, 5, 5, activation='relu', subsample = (2, 2), border_mode = 'same', init="glorot_uniform", bias = True))

model.add(Convolution2D(64, 3, 3, activation='relu', border_mode = 'same', init="glorot_uniform", bias = True))

model.add(Convolution2D(64, 3, 3, activation='relu', border_mode = 'same', init="glorot_uniform", bias = True))

model.add(Flatten())

# Next, five fully connected layers
model.add(Dense(1164, activation='relu', init="glorot_uniform", bias = True))
model.add(Dense(100, activation='relu', init="glorot_uniform", bias = True))
model.add(Dense(50, activation='relu', init="glorot_uniform", bias = True))
model.add(Dense(10, activation='relu', init="glorot_uniform", bias = True))
model.add(Dense(1, activation='relu', init="glorot_uniform", bias = True))

model.summary()

# save the model achitecture
json_string = model.to_json()
with open(mode_path+'model.json', 'w') as json_file:
    json_file.write(json_string)

print('model saved as json')
