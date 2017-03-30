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
import random
from sklearn.utils import shuffle

# folder name, where the data is given
data_labels_path = './speed_challenge/drive.json'
data_extracted_path = './data_extracted/'

#split the train and validation data
split_train_validation = 25 #represnts the test data %

# load the json data
with open(data_labels_path) as data_file:
   data = json.load(data_file)

# split the data
train=[]
val=[]

#shuffle the data
def shuffleData(mode="rgb"):
    global train
    global val
    for i in range(0, len(data)-1):
        rand_idx = np.random.randint(len(data)-1)
        randInt = np.random.randint(100)
        if randInt <= split_train_validation:
            #update the validation data
            val.append([data[rand_idx][0], data[rand_idx][1]])
            if mode == "dense_optical_flow":
                # update in pairs, as we do motion analysis
                val.append([data[rand_idx+1][0], data[rand_idx+1][1]])
        else:
            #update training dat
            train.append([data[rand_idx][0], data[rand_idx][1]])
            if mode == "dense_optical_flow":
                train.append([data[rand_idx+1][0], data[rand_idx+1][1]])

    print("train size: %d" % len(train))
    print("val size %d "  % len(val))


# crops the image
def crop(image, top_percent, bottom_percent, left_percent, right_percent):

    top = int(np.ceil(image.shape[0] * top_percent))
    bottom = image.shape[0] - int(np.ceil(image.shape[0] * bottom_percent))
    left = int(np.ceil(image.shape[1] * left_percent))
    right = image.shape[1] - int(np.ceil(image.shape[1] * right_percent))

    return image[top:bottom, left:right] #image[100:440, :-90]

# resizes the image
def resize(image, new_dim):
    return cv2.resize(image, new_dim, interpolation = cv2.INTER_AREA)

# calculates the magnitude and angle of of the optical flow vectors (u, v),
# based on Gunner Farneback's algorithm, calculates optical flow for all the pixels
### took reference from http://docs.opencv.org/3.1.0/d7/d8b/tutorial_py_lucas_kanade.html
def getOpticalFlowDense(image, next_image, vis_optical_flow=False):
    # convert to grey scale
    image_grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    next_image_grey = cv2.cvtColor(next_image, cv2.COLOR_RGB2GRAY)

    # compute the optical flow dense
    flow = cv2.calcOpticalFlowFarneback(image_grey, next_image_grey, None, 0.5, 1, 15, 2, 5, 1.3, 0)

    # get the magnitude and angle of the optical flow vectors
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])

    # hsv mask with the image size
    hsv = np.zeros_like(image)

    # set saturation
    hsv[:,:,1] = cv2.cvtColor(next_image, cv2.COLOR_RGB2HSV)[:,:,1]

    # optical flow vector angle in hue
    hsv[...,0] = ang*180/np.pi/2

    # optical flow vector mahnitude is in value
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)

    # convert back to RGB
    hsv = np.asarray(hsv, dtype= np.float32)
    rgb_optical_flow_dense = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)

    # viusulisation of dense optical flow
    if vis_optical_flow == True:
        cv2.imshow('optical_flow',rgb_optical_flow_dense)
        cv2.waitKey(0)

    return rgb_optical_flow_dense


# preprocess the images before training
def processGeneratedImage(image, factor=1, resize_dim=(220, 66)):

    #convert to rgb
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # crop the image
    # crop the sky, right side black part and the bottom car symbol
    image_crop = crop(image, 0.2, 0.08, 0, 1.14)

    # resize
    image = resize(image_crop, resize_dim)

    return image

# fetch the randmomised images and theri labels
def fetchRandomizedImages_Labels(mode, batch_size=64):
    if mode == 'train':
        n_img = len(train)
    if mode == 'val':
        n_img = len(val)

    rnd_indices = np.random.randint(0, n_img, batch_size)
    x_y = []
    for i in rnd_indices:
        if mode == 'train':
            x_y.append((data_extracted_path+"%f.jpg" % train[i][0], train[i][1]))
        if mode == 'val':
            x_y.append((data_extracted_path+"%f.jpg" % val[i][0], val[i][1]))
    return x_y

# generator yields next training set
def genData(mode, batch_size=64):
    while True:
        X_batch = []
        y_batch = []

        if mode == 'train':
            images = fetchRandomizedImages_Labels('train', batch_size)
        if mode == 'val':
            images = fetchRandomizedImages_Labels('val', batch_size)

        for img_file, speed in images:
            raw_image = plt.imread(img_file)
            new_image = processGeneratedImage(raw_image)
            X_batch.append(new_image)
            y_batch.append(speed)

        # shuffle tha data and yield
        yield shuffle(np.array(X_batch), np.array(y_batch))

# generator yields next training set, using dense optical flow
def genDataDOpticalflow(mode, batch_size=64):
    while True:
        X_batch = []
        y_batch = []
        for i in range(0, batch_size):
            if mode == 'train':
                # generate a random number
                rand_num = np.random.randint(1, len(train)-1) # make sure we dont wrap around the bounds

                # get 3 images and speeds around the random index, and decide which is current and which is next
                prev_idx = train[rand_num - 1][0]
                speed_prev = train[rand_num - 1][1]

                curr_idx = train[rand_num][0]
                speed_curr = train[rand_num][1]

                next_idx = train[rand_num + 1][0]
                speed_next = train[rand_num + 1][1]

                # sort the timings as the data is shuffled
                sort_idx=[[prev_idx, speed_prev], [curr_idx, speed_curr], [next_idx, speed_next]]
                sort_idx.sort()

                # as we need frames next to each other
                if sort_idx[1][0] - sort_idx[0][0] < 0.4:
                    curr_img_idx = sort_idx[0][0]
                    next_img_idx = sort_idx[1][0]
                    curr_speed = sort_idx[0][1]
                    next_speed = sort_idx[1][1]
                elif sort_idx[2][0] - sort_idx[1][0] < 0.4:
                    curr_img_idx = sort_idx[1][0]
                    next_img_idx = sort_idx[2][0]
                    curr_speed = sort_idx[1][1]
                    next_speed = sort_idx[2][1]
                else:
                    print("bummer..!!!!!, eliminate the sample")

            if mode == 'val':
                # generate a random number
                rand_num = np.random.randint(0, len(val)-1)

                # get 3 images and speeds around the random index, and decide which is current and which is next
                prev_idx = val[rand_num - 1][0]
                speed_prev = val[rand_num - 1][1]

                curr_idx = val[rand_num][0]
                speed_curr = val[rand_num][1]

                next_idx = val[rand_num + 1][0]
                speed_next = val[rand_num+1][1]

                # sort the timings as the data is shuffled
                sort_idx=[[prev_idx, speed_prev], [curr_idx, speed_curr], [next_idx, speed_next]]
                sort_idx.sort()

                # as we need frames next to each other
                if sort_idx[1][0] - sort_idx[0][0] < 0.4:
                    curr_img_idx = sort_idx[0][0]
                    next_img_idx = sort_idx[1][0]
                    curr_speed = sort_idx[0][1]
                    next_speed = sort_idx[1][1]
                elif sort_idx[2][0] - sort_idx[1][0] < 0.4:
                    curr_img_idx = sort_idx[1][0]
                    next_img_idx = sort_idx[2][0]
                    curr_speed = sort_idx[1][1]
                    next_speed = sort_idx[2][1]
                else:
                    print("bummer..!!!!!, eliminate the sample")

            # process the image
            b_factor = 0.2 + np.random.uniform()

            curr_img = plt.imread(data_extracted_path+"%f.jpg" % curr_img_idx)
            curr_img = processGeneratedImage(curr_img, b_factor)
            next_img = plt.imread(data_extracted_path+"%f.jpg" % next_img_idx)
            next_img = processGeneratedImage(next_img, b_factor)

            # get the dense flow of the 2 images
            rgb_dense_flow = getOpticalFlowDense(curr_img, next_img)

            # append the data
            X_batch.append(rgb_dense_flow)

            # get the mean speed from both the images
            speed=np.mean([curr_speed, next_speed])
            y_batch.append(speed)

        yield shuffle(np.array(X_batch), np.array(y_batch))

# this returns the images as a numpy array
def load_xInput():
    x=[]
    for i in range(0, len(data)):
        xt=plt.imread(data_extracted_path+"%f.jpg" % data[i][0])
        raw_speed = data[i][1]
        new_image = processGeneratedImage(xt)
        x.append(new_image)

    return np.array(x)

# this returns the speeds as a numpy array
def load_yLabels():
    y=[]
    for i in range(0, len(data)):
        yt=data[i][1]
        y.append(yt)

    return np.array(y)

def load_XDenseOptFlowInput():
    x=[]
    for i in range(0, len(data)-1):
        # get the current image and the next image
        curr_img = data[i][0]
        next_img = data[i + 1][0]

        # process the image
        curr_img = plt.imread(data_extracted_path+"%f.jpg" % curr_img)
        curr_img=processGeneratedImage(curr_img)
        next_img = plt.imread(data_extracted_path+"%f.jpg" % next_img)
        next_img=processGeneratedImage(next_img)

        # get the dense flow of the 2 images
        rgb_dense_flow = getOpticalFlowDense(curr_img, next_img)

        # append the data
        x.append(rgb_dense_flow)

    return np.array(x)

# this returns the speeds as a numpy array
def load_yDenseOptFlowLabels():
    y=[]
    for i in range(1, len(data)):
        # as we estimate the mean of the speed of both the images
        yt = np.mean([data[i-1][1], data[i][1]])
        y.append(yt)

    return np.array(y)
