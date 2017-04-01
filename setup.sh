#!/bin/bash
## Usage: bash setup.sh
## Author: sumanth
## Purpose: setups the system with miniconda and env variable
##
## Options:
##   none
##

###
# based on https://github.com/udacity/CarND-Term1-Starter-Kit)

# get this in home
cd

#get the miniconda
echo "$(tput setaf 2)Downloading Miniconda$(tput sgr0)"
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh

#setup miniconda
echo "$(tput setaf 2)setup Miniconda$(tput sgr0)"
bash Miniconda3-latest-Linux-x86_64.sh

#delete this
rm -rf Miniconda3-latest-Linux-x86_64.sh

#clone the udacity repo:
git clone https://github.com/udacity/CarND-Term1-Starter-Kit.git

#create the environment variable
echo "$(tput setaf 2)Setting up the environment variable$(tput sgr0)"
export PATH="$HOME/miniconda3/bin:$PATH"
cd CarND-Term1-Starter-Kit
conda env create -f environment.yml
conda clean -tp
