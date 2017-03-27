#!/usr/bin/python
## Author: sumanth
## Date: March 26, 2017
# infers the trained model

# Python 2/3 compatibility
from __future__ import print_function

from keras.models import model_from_json
import load_data
import numpy
import matplotlib.pyplot as plt
import time

# weights path
model_path='./model_weights/model.json'
model_weights_path='./model_weights/weights.h5'

json_file = open(model_path, 'r')
loaded_model_val = json_file.read()
json_file.close()
model_val = model_from_json(loaded_model_val)
print("Loaded the model")

# load the trained weights
model_val.load_weights(model_weights_path)
print("trained weights loaded")

# compile the model
model_val.compile(loss='mse', optimizer='adam')
print("compiled the model")

# get the values of speed and the images
y_actual = load_data.load_yLabels()
x = load_data.load_xInput()

y_predicted=model_val.predict(x)
print("output predicted")
print(len(y_predicted))

#Plotting speed actual vs predicted
plt.figure(0)
plt.plot(y_actual, label = 'Actual Dataset')
plt.plot(y_predicted, label = 'Training Prediction')
plt.title('speed: Actual vs Predicted')
plt.xlabel('Number of images')
plt.ylabel('speed')
plt.legend(loc = 'upper left')
plt.savefig('speed predicted optical flow')
print("Saved speed plot to disk")
plt.close()
