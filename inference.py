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
import argparse
import scipy

# path for the predicted images
data_output_path='./data_predicted/'

# weights path
model_path='./model_weights/model.json'
model_weights_path='./model_weights/'

# optical flow dense draw on the video
# reference from here: https://github.com/opencv/opencv/blob/master/samples/python/opt_flow.py
def drawDenseOptFlow(image, next_image):
    # convert to grey scale
    image_grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    next_image_grey = cv2.cvtColor(next_image, cv2.COLOR_RGB2GRAY)

    # compute the optical flow dense
    flow = cv2.calcOpticalFlowFarneback(image_grey, next_image_grey, None, 0.5, 1, 15, 2, 5, 1.3, 0)

    # draw the flow
    step=16
    h, w = next_image_grey.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 2.5)
    vis_flow = cv2.cvtColor(next_image_grey, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis_flow, lines, 1, (0, 0, 255), 2)
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis_flow, (x1, y1), 2, (0, 255, 0), -1)

    # draw the hsv
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*4, 255)
    vis_hsv = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

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
    vis_hsv_rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)

    return vis_flow, vis_hsv, vis_hsv_rgb

def main(images_extracted_path, data_json_path, mode="dense_optical_flow", video_generation="no"):

    json_file = open(model_path, 'r')
    loaded_model_val = json_file.read()
    json_file.close()
    model_val = model_from_json(loaded_model_val)
    print("Loaded the model")

    # load the trained weights
    if mode == "dense_optical_flow":
        model_val.load_weights(model_weights_path+'DenseOptflow_weights.h5')
    else:
        model_val.load_weights(model_weights_path+'rgb_weights.h5')
    print("trained weights loaded")

    # compile the model
    model_val.compile(loss='mse', optimizer='adam')
    print("compiled the model")

    # load the json data, with image names and the speeds
    with open(data_json_path) as data_file:
        data = json.load(data_file)

    if mode == "dense_optical_flow":
        x = load_data.load_XDenseOptFlowInput(data)
        y_actual = load_data.load_yDenseOptFlowLabels(data)
    else:
        x = load_data.load_xInput(data)
        y_actual = load_data.load_yLabels(data)

    # evaluate the weights
    print('evaluating....')
    score = model_val.evaluate(x, y_actual, 64, verbose=1)
    print('Evaluation loss: %f' % score)

    print('predicting.....')
    y=model_val.predict(x)
    y=y.flatten()
    y_predicted=y

    # smoothing the predicted data, can be removed
    if mode == "dense_optical_flow":
        # smooth using Savitzky–Golay filter
        y_predicted = scipy.signal.savgol_filter(y, 101, 3) # 101 window length and fit using 3 order polynomial
    else:
        # smooth using Savitzky–Golay filter
        y_predicted = scipy.signal.savgol_filter(y, 51, 3) # 51 window length and fit using 3 order polynomial

    # Plotting speed actual vs predicted
    plt.figure(0)
    #plt.plot(y, label = 'Training Prediction')
    plt.plot(y_predicted, label = 'Training Prediction smoothed')
    plt.plot(y_actual, label = 'Actual Dataset')
    plt.title('speed: Actual vs Predicted')
    plt.xlabel('Number of images')
    plt.ylabel('speed')
    plt.legend(loc = 'upper left')
    if mode == "dense_optical_flow":
        plt.savefig('speed predicted optical flow')
    else:
        plt.savefig('speed predicted rgb')
    print("Saved speed plot to disk")
    plt.close()

    # annotate the images and save
    if video_generation == "yes":
        print('generating video.....')

        for i in range(0, len(y_predicted)):
            if mode == "dense_optical_flow":
                cur_im = cv2.imread(images_extracted_path+"%f.jpg" % data[i][0])
                nxt_im = cv2.imread(images_extracted_path+"%f.jpg" % data[i+1][0])

                xt=nxt_im.copy()
                cv2.putText(xt,'Act Speed = ' + str(y_actual[i]), (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
                cv2.putText(xt,'Pred Speed = ' + str(y_predicted[i]), (10,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
                error=y_actual[i]-y_predicted[i]
                cv2.putText(xt,'Error = ' + str(error), (10,120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

                # get the visulaoisations for the flow
                vis_flow, vis_hsv, vis_rgb_hsv = drawDenseOptFlow(cur_im, nxt_im)

                # merge all the vis
                merged1 = np.hstack((xt,vis_flow))
                merged2 = np.hstack((vis_hsv,vis_rgb_hsv))
                merged = np.vstack((merged1,merged2))

                # save the images
                cv2.imwrite(data_output_path+"%i.jpg" % i, merged)
            else:
                xt=cv2.imread(images_extracted_path+"%f.jpg" % data[i][0])
                cv2.putText(xt,'Act Speed = ' + str(y_actual[i]), (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
                cv2.putText(xt,'Pred Speed = ' + str(y_predicted[i]), (10,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
                error=y_actual[i]-y_predicted[i]
                cv2.putText(xt,'Error = ' + str(error), (10,120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

                # save the images
                cv2.imwrite(data_output_path+"%i.jpg" % i, xt)

            if i%1000 == 0:
                print("%d frames processed" % i)

        # generate the video, from the predicted annotated images
        vimages = [data_output_path+"%d.jpg" % i for i in range(0, len(y_predicted))]
        clip = ImageSequenceClip(vimages, fps=25)
        if mode == "dense_optical_flow":
            clip.write_videofile('speed_predicted_optDense_flow.mp4', fps=25)
        else:
            clip.write_videofile('speed_predicted_rgb.mp4', fps=25)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='inference for speed estimation')
    parser.add_argument(
        '--extracted_images',
        type=str,
        default='./data_extracted/',
        help='path for the extarcted images'
    )
    parser.add_argument(
        '--data_json',
        type=str,
        default="./speed_challenge/drive.json",
        help='path for the json file, which has the image names and the speeds'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default="dense_optical_flow",
        help='mode to indicate whether the inference should happen with rgb or dense optical flow'
    )
    parser.add_argument(
        '--video',
        type=str,
        default="yes",
        help='flag to indicate whether the video should be generated'
    )
    args = parser.parse_args()

    main(args.extracted_images, args.data_json, args.mode, args.video,)
