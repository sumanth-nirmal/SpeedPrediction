#!/usr/bin/python
## Author: sumanth
## Date: March 26, 2017
# Estimate the speed of car from dash board capera images based on optical flow

## the reference from this is from:
## https://github.com/opencv/opencv/blob/master/samples/python/lk_track.py

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2
from time import clock
import json
import argparse

# path for the data input and the data labels, change if required
data_video_path='./speed_challenge/drive.mp4'
data_label_path='./speed_challenge/drive.json'

# crops the image
def crop(image, top_percent, bottom_percent, left_percent, right_percent):

    top = int(np.ceil(image.shape[0] * top_percent))
    bottom = image.shape[0] - int(np.ceil(image.shape[0] * bottom_percent))
    left = int(np.ceil(image.shape[1] * left_percent))
    right = image.shape[1] - int(np.ceil(image.shape[1] * right_percent))

    return image[top:bottom, left:right]

# canny edge detector
def cannyMask(image):
    mask = cv2.Canny(image, 100, 300)
    cv2.imshow('mask', mask)
    cv2.waitKey(1)
    return mask

def estimateSpeed(cap, vis_enable="no"):
    track_len = 10
    detect_interval = 5
    frame_idx = 0
    tracks = []

    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 500,
                           qualityLevel = 0.3,
                           minDistance = 7,
                           blockSize = 7 )

    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15, 15),
                      maxLevel = 2,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # load the true measured speed and the time stamps
    data_labels = np.array(json.load(open(data_label_path)))
    time_stamp=data_labels[:, 0]
    speed_measured=data_labels[:,1]

    # if the video capture is sucessfully opened
    while cap.isOpened():
            # get the frame
            ret, frame = cap.read()

            # crop the image
            frame = crop(frame, 0.5, 0.05, 0.2, 0.1)

            # convert to grey scale
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            vis = frame.copy()

            if len(tracks) > 0:
                img0, img1 = prev_gray, frame_gray
                p0 = np.float32([tr[-1] for tr in tracks]).reshape(-1, 1, 2)
                p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)

                # sanity check
                d = abs(p0-p0r).reshape(-1, 2).max(-1)
                good = d < 1

                new_tracks = []
                for tr, (x, y), good_flag in zip(tracks, p1.reshape(-1, 2), good):
                    if not good_flag:
                        continue
                    tr.append((x, y))
                    if len(tr) > track_len:
                        del tr[0]
                    new_tracks.append(tr)
                    cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
                tracks = new_tracks
                cv2.polylines(vis, [np.int32(tr) for tr in tracks], False, (0, 255, 0))

                print(p1)
                cv2.imshow('optical_flow', vis)
                cv2.waitKey(1)

            if frame_idx % detect_interval == 0:
                mask = np.zeros_like(frame_gray)
                mask[:] = 255
                for x, y in [np.int32(tr[-1]) for tr in tracks]:
                    cv2.circle(mask, (x, y), 5, 0, -1)
                p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
                if p is not None:
                    for x, y in np.float32(p).reshape(-1, 2):
                        tracks.append([(x, y)])

            frame_idx += 1
            prev_gray = frame_gray
            if vis_enable == "yes":
                cv2.imshow('optical_flow', vis)
                ch = cv2.waitKey(1)
                if ch == 27:
                    break

def main(vis):
    # load the input video
    cap = cv2.VideoCapture(data_video_path)
    if cap.isOpened():
        estimateSpeed(cap, vis)
    else:
        print("cant open the video..!!!!!!!!!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training for speed estimation')
    parser.add_argument(
        '--vis',
        type=str,
        help='yes if visualisation is required '
    )
    args = parser.parse_args()

    main(args.vis)
