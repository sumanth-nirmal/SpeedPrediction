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

#load the json data
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


def resize(image, new_dim):
    """
    Resize a given image according the the new dimension
    :param image:
        Source image
    :param new_dim:
        A tuple which represents the resize dimension
    :return:
        Resize image
    """
    return scipy.misc.imresize(image, new_dim)

def processGenerateImagespeed(image, speed, top_crop_percent=0.35, bottom_crop_percent=0.1,
                       resize_dim=(64, 64), do_shear_prob=0.9):

    """
    Applies all the image procssing and returns the steering angle with the image

    :param image:
    :param speed:
    :param top_crop_percent:
    :param bottom_crop_percent:
    :param resize_dim:
    :param do_shear_prob:
    :param shear_range:
    :return:
    """


    # shear image
    #if(bernoulli.rvs(do_shear_prob)):
    #    image, speed = randomShear(image, speed)

    # crop image
    #image = crop(image, top_crop_percent, bottom_crop_percent)

    # flip the image
    #image, speed = randomFlip(image, speed)

    # random gamma
    #image = randomGamma(image)

    # resize
    image = resize(image, resize_dim)

    return image, speed

def fetchImages(batch_size=64):
    """
    :param batch_size:
        Size of the image batch
    :return:
        An list of selected (image files names, corresponding speed)
    """
    num_of_img = len(data)
    rnd_indices = np.random.randint(0, num_of_img, batch_size)
    image_files_and_speeds = []
    for i in rnd_indices:
            image_files_and_speeds.append((data_extracted_path+"%f.jpg" % data[i][0], data[i][1]))

    return image_files_and_speeds


def genBatch(batch_size=64):
    """
    This is a generator which yields the next training batch
    :param batch_size:
        Number of training images in a single batch
    :return:
        A tuple of features and speeds as two numpy arrays
    """
    while True:
        X_batch = []
        y_batch = []
        images = fetchImages(batch_size)
        for img_file, speed in images:
            raw_image = plt.imread(img_file)
            raw_speed = speed
            new_image, new_speed = processGenerateImagespeed(raw_image, raw_speed)
            X_batch.append(new_image)
            y_batch.append(new_speed)

        assert len(X_batch) == batch_size, 'len(X_batch) == batch_size should be True'
        yield np.array(X_batch), np.array(y_batch)

def genVel(batch_size=64):
    """
    This is a generator which yields the next training batch
    :param batch_size:
        Number of training images in a single batch
    :return:
        A tuple of features and speeds as two numpy arrays
    """
    while True:
        X_batch = []
        for i in range(0,batch_size):
            raw_image = plt.imread(data_extracted_path+"%f.jpg" % data[i][0])
            raw_speed = data[i][1]
            new_image, new_speed = processGenerateImagespeed(raw_image, raw_speed)
            X_batch.append(new_image)

        yield np.array(X_batch)

def dataGen(batch_size=64):
    while True:
        x=[]
        y=[]
        for i in range(0,batch_size):
            xt=cv2.imread(data_extracted_path+"%f.jpg" % data[i][0])
            #print (xt.shape)
            xt=cv2.resize(xt, (64, 64)).astype(np.float32)
            #print (xt.shape)
            x.append(xt)
            y.append(data[i][1])
            return x, y

# this returns the images as a numpy array
def load_x():
    x=[]
    for i in range(0,len(data)):
        xt=plt.imread(data_extracted_path+"%f.jpg" % data[i][0])
        raw_speed = data[i][1]
        new_image, new_speed = processGenerateImagespeed(xt, raw_speed)
        x.append(new_image)

    return np.array(x)

# this returns the speeds as a numpy array
def load_speed():
    y=[]
    for i in range(0,len(data)):
        yt=data[i][1]
        y.append(yt)

    return np.array(y)
