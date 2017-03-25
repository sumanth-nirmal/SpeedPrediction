#!/usr/bin/python
## Author: sumanth
## Date: March 23, 2017
# loads the data from json file

import json
import cv2

#load the json data
with open('drive.json') as data_file:
   data = json.load(data_file)

def load_data():
    x=[]
    y=[]
    for i in range(0,len(data)):
        x.append(cv2.imread("./data/%f.jpg" % data[i][0]))
        y.append(data[i][1])
    return x, y
