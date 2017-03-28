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
import json
import cv2
import numpy as np
from moviepy.editor import ImageSequenceClip

# weights path
model_path='./model_weights/model.json'
model_weights_path='./model_weights/weights.h5'

data_labels_path='./speed_challenge/drive.json'

data_path='./data_extracted/'
data_output_path='./data_predicted/'
output_file = 'speed_predicted_output.mp4'

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

# evaluate the weights
print('evaluating....')
score = model_val.evaluate(x, y_actual, 64, verbose=1)
print('Evaluation loss: %f' % score)

print('predicting.....')
y_predicted=model_val.predict(x)
print("output predicted")

# Plotting speed actual vs predicted
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

# annotate the images and save
print('generating video.....')
# load the json data
with open(data_labels_path) as data_file:
   data = json.load(data_file)

for i in range(0, len(y_predicted)):
    xt=cv2.imread(data_path+"%f.jpg" % data[i][0])
    cv2.putText(xt,'Act Speed = ' + str(y_actual[i]), (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
    cv2.putText(xt,'Pred Speed = ' + str(y_predicted[i]), (10,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
    error=y_actual[i]-y_predicted[i]
    cv2.putText(xt,'Error = ' + str(error), (10,120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    if i%1000 == 0:
        print("%d frames processed" % i)
    cv2.imwrite(data_output_path+"%i.jpg" % i, xt)

# generate the video, from the predicted annotated images
vimages = [data_output_path+"%d.jpg" % i for i in range(0, len(data))]
clip = ImageSequenceClip(vimages, fps=13)
clip.write_videofile(output_file, fps=13)
