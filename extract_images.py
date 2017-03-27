#!/usr/bin/python
## Author: sumanth
## Date: March 23, 2017
#extract images from video

# Python 2/3 compatibility
from __future__ import print_function

import cv2
import json

# folder name, where the data is given
data_labels_path = './speed_challenge/drive.json'
data_path = './speed_challenge/drive.mp4'
data_extracted_path = './data_extracted/'

#load the json data
with open(data_labels_path) as data_file:
   data = json.load(data_file)

# load the data
cap = cv2.VideoCapture(data_path)
count = 0
while cap.isOpened():
    ret,frame = cap.read()
    cv2.imshow('window-name',frame)
    cv2.imwrite(data_extracted_path+"%f.jpg" % data[count][0], frame)
    count = count + 1
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break


cap.release()
