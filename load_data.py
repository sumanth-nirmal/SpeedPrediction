#!/usr/bin/python
## Author: sumanth
## Date: March 23, 2017
# loads the data from json file and has helper functions

# Python 2/3 compatibility
from __future__ import print_function

import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
from scipy.ndimage import rotate
from scipy.stats import bernoulli

# folder name, where the data is given
data_labels_path = './speed_challenge/drive.json'
data_extracted_path = './data_extracted/'

# load the json data
with open(data_labels_path) as data_file:
   data = json.load(data_file)

def crop(image, top_percent, bottom_percent):
    """
    Crops an image according to the given parameters
    :param image: source image
    :param top_percent:
        The percentage of the original image will be cropped from the top of the image
    :param bottom_percent:
        The percentage of the original image will be cropped from the bottom of the image
    :return:
        The cropped image
    """
    assert 0 <= top_percent < 0.5, 'top_percent should be between 0.0 and 0.5'
    assert 0 <= bottom_percent < 0.5, 'top_percent should be between 0.0 and 0.5'

    top = int(np.ceil(image.shape[0] * top_percent))
    bottom = image.shape[0] - int(np.ceil(image.shape[0] * bottom_percent))

    return image[top:bottom, :]

# resizes the image
def resize(image, new_dim):
    return scipy.misc.imresize(image, new_dim)

# preprocess the images before training
def processGeneratedImage(image, speed, resize_dim=(64, 64)):

    # resize
    image = resize(image, resize_dim)

    return image, speed

# fetch the randmomised images and theri labels
def fetchRandomizedImages_Labels(batch_size=64):
    n_img = len(data)
    rnd_indices = np.random.randint(0, n_img, batch_size)
    x_y = []
    for i in rnd_indices:
            x_y.append((data_extracted_path+"%f.jpg" % data[i][0], data[i][1]))

    return x_y

# generator yields next training set
def genData(batch_size=64):
    while True:
        X_batch = []
        y_batch = []
        images = fetchRandomizedImages_Labels(batch_size)
        for img_file, speed in images:
            raw_image = plt.imread(img_file)
            raw_speed = speed
            new_image, new_speed = processGeneratedImage(raw_image, raw_speed)
            X_batch.append(new_image)
            y_batch.append(new_speed)

        yield np.array(X_batch), np.array(y_batch)

# this returns the images as a numpy array
def load_xInput():
    x=[]
    for i in range(0,len(data)):
        xt=plt.imread(data_extracted_path+"%f.jpg" % data[i][0])
        raw_speed = data[i][1]
        new_image, new_speed = processGeneratedImage(xt, raw_speed)
        x.append(new_image)

    return np.array(x)

# this returns the speeds as a numpy array
def load_yLabels():
    y=[]
    for i in range(0,len(data)):
        yt=data[i][1]
        y.append(yt)

    return np.array(y)
