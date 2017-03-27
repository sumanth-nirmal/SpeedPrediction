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

json_file = open('./model_weights/model.json', 'r')
loaded_model_val = json_file.read()
json_file.close()
model_val = model_from_json(loaded_model_val)
print("Loaded the model")

#Load the trained weights
model_val.load_weights("./model_weights/weights" + ".h5")
print("trained weights loaded")

# compile the model
model_val.compile(loss='mse', optimizer='adam')
print("compiled the model")

# get the values of speed and the images
y_train_data = load_data.load_speed()
x= load_data.load_x()
#genT = load_data.genVel('center')

#trainPredict = model_val.predict_generator(genT, val_samples = len(load_data.data))
y=model_val.predict(x)
print("output predicted")
print(len(y))
print(y)
print(y_train_data)

# shift train predictions for plotting
# trainPredictPlot = numpy.empty_like(y_train_data)
# trainPredictPlot[:, :] = numpy.nan
# trainPredictPlot[0:len(trainPredict), :] = trainPredict

#Plotting steering angle actual vs predicted
plt.figure(0)
plt.plot(y_train_data, label = 'Actual Dataset')
plt.plot(y, label = 'Training Prediction')
plt.title('speed: Actual vs Predicted')
plt.xlabel('Number of images')
plt.ylabel('speed')
plt.legend(loc = 'upper left')
plt.savefig('speed predicted')
print("Saved speed plot to disk")
plt.close()
