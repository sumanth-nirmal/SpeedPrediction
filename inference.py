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
from moviepy.editor import VideoFileClip

# weights path
model_path='./model_weights/model.json'
model_weights_path='./model_weights/weights.h5'

data_path='./data_extracted/'
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

# annotate the images and generate the video
print('generating video.....')
for i in range(0, len(y_predicted)):
    xt=plt.imread(data_path+"%f.jpg" % data[i][0])
    cv2.putText(xt,'Actual Speed = {:1.2}'.format(y_actual[i]), (np.int(cols/2)-100,50), font, 1,(255,255,255),2)
    cv2.putText(xt,'Predicted Speed = {:.0f}'.format(y_predicted[i]), (np.int(cols/2)-100,100), font, 1,(255,255,255),2)
    error=y_actual[i]-y_predicted[i]
    cv2.putText(xt,'Error = {:.0f}'.format(error), (np.int(cols/2)-100,150), font, 1,(255,255,255),2)
    annotated_video = video.fl_image(lambda img: xt)
    annotated_video.write_videofile(output_file, audio=False)
