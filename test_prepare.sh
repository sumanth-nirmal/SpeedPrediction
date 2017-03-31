#!/bin/bash
## Usage: bash test.sh
## Author: sumanth
## Date: March 30, 2017
## Purpose: test shell script for preparing the setup
##
## Options:
##   none


#set the directories
mkdir ./data_extracted/
mkdir ./data_predicted/

rm -rf ./data_extracted/*
rm -rf ./data_predicted/*


echo "$(tput setaf 2)Setup done$(tput sgr 0)"
