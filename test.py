#!/bin/bash
## Usage: bash test.sh
## Author: sumanth
## Date: March 30, 2017
## Purpose: test shell script for testing the speed estimatiobn based on the video
##
## Options:
##   none


## Usage
# bash test.sh <path to video> <path to data json file> <mode> <video>
# path to the video, ex: "./speed_challenge/drive.mp4"
# path to the data json file, ex: "./speed_challenge/drive.json"
# mode, ex: "dense_optical_flow" or "rgb"
# video ex: "yes" or "no"

import subprocess
import extract_images
import inference

# prepare the test setup
print("preparing the setup")
subprocess.call['./test_prepare.sh']

# etxract images from the video
print("extracting the images")
extract_images.main("./speed_challenge/drive.mp4", "./speed_challenge/drive.json")

# predict the steering angles and generate the video
print("predict the speed")
inference.main("./data_extracted/", "./speed_challenge/drive.json", "dense_optical_flow", "yes")
