#!/usr/bin/python
## Author: sumanth
## Date: March 23, 2017
# model to train the data

# Python 2/3 compatibility
from __future__ import print_function

from keras.models import model_from_json
import load_data
import numpy
import matplotlib.pyplot as plt
from keras.optimizers import Adam
import time
import argparse

# file path for the model
model_path = './model_weights/model.json'
weights_path = './model_weights'

# training parameters
number_of_epochs = 10
learning_rate = 1e-4

def main(mode="dense_optical_flow"):
    # load json model
    json_file = open(model_path, 'r')
    loaded_model = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model)
    print ("Loaded the training model")

    # complie the model
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam, loss="mse")

    # create two generators for training and validation
    if mode == "dense_optical_flow":
        print("training using optical dense flow")
        # create two generators for training and validation
        trainGen = load_data.genDataDOpticalflow('train', 16)
        valGen = load_data.genDataDOpticalflow('val')
        #!!!!!!!!! same gnerator as val
        evalGen = load_data.genDataDOpticalflow('val')
    else:
        print("training using rgb")
        trainGen = load_data.genData('train', 64)
        valGen = load_data.genData('val')
        #!!!!!!!!! same gnerator as val
        evalGen = load_data.genData('val')

    # train the model
    history = model.fit_generator(trainGen,
                              samples_per_epoch=len(load_data.train),
                              nb_epoch=number_of_epochs,
                              validation_data=valGen,
                              nb_val_samples=len(load_data.val),
                              verbose=1)

    # evaluate the model
    score = model.evaluate_generator(evalGen, 1250, max_q_size=10)
    print('Evaluation loss: %f' % score)

    if mode == "dense_optical_flow":
        # save the weights
        model.save_weights(weights_path+'/DenseOptflow_weights.h5')
        #save the model with weights
        model.save(weights_path+'/DenseOptflow_model_weights.h5')
    else:
        # save the weights
        model.save_weights(weights_path+'/rgb_weights.h5')
        #save the model with weights
        model.save(weights_path+'/rgb_model_weights.h5')

    # plots
    # summarize history for loss
    plt.figure(1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['training', 'validation'], loc='upper left')
    if mode == "dense_optical_flow":
        plt.savefig('speed_prediction_loss_plot_denseOptFlow')
    else:
        plt.savefig('speed_prediction_loss_plot_rgb')
    print("Saved loss plot to disk")
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training for speed estimation')
    parser.add_argument(
        '--mode',
        type=str,
        help='mode to indicate training either with rgb or with dense optical flow\n usage:python train.py --mode dense_optical_flow or \n python train.py --mode rgb'
    )
    args = parser.parse_args()

    #shuffle the data
    load_data.shuffleData(args.mode)
    main(args.mode)
