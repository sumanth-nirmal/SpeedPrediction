#!/usr/bin/python
## Author: sumanth
## Date: March 23, 2017
#extract images from video

# Python 2/3 compatibility
from __future__ import print_function

import cv2
import json
import argparse
import subprocess

# no.of images extracted
count=0

data_extracted_path='./data_extrated/'
def main(video_path, data_json_path):
    # if data json file is not avilable
    if data_json_path != "no":
        #load the json data
        with open(data_json_path) as data_file:
            data = json.load(data_file)

    # load the data
    cap = cv2.VideoCapture(video_path)

    # check if the video is opened sucessfully
    if cap.isOpened() == True:
        global count

        # frames per sec
        fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
        print("Frames per second in the input video: {0}".format(fps))

        while cap.isOpened():
            ret,frame = cap.read()
            cv2.imshow('video',frame)
            if data_json_path != "no":
                cv2.imwrite(data_extracted_path+"%f.jpg" % data[count][0], frame)
            else:
                cv2.imwrite(data_extracted_path+"%f.jpg" % count, frame)
            count = count + 1
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
    else:
        print("the video can not be opened\ncheck the video or use different opencv version\nusing opencv version: ",cv2.__version__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='extract images from the video and labels if avialable')
    parser.add_argument(
        '--video',
        type=str,
        help='path for the video, usage: --video "./speed_challenge/drive.mp4"'
    )
    parser.add_argument(
        '--data_json',
        type=str,
        help='path for the json file if available, usage: --data_json "./speed_challenge/drive.json" or --data_json "no" if not avialable'
    )
    args = parser.parse_args()

    main(args.video, args.data_json)
