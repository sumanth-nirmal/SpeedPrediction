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

number_of_epochs = 50
number_of_samples_per_epoch = 120
number_of_validation_samples = 432
learning_rate = 1e-4

# file path for the model
model_path = './model_weights/model.json'
weights_path = './model_weights'

# load json model
json_file = open(model_path, 'r')
loaded_model = json_file.read()
json_file.close()
model = model_from_json(loaded_model)
print ("Loaded the training model")

# complie the model
model.compile(optimizer=Adam(learning_rate), loss="mse", )

#get the data
# x, y = load_data.load_data()
# print(len(x))
# print(len(y))
# print("data loaded")

# train the model
#history = model.fit(x,y, batch_size=number_of_samples_per_epoch, nb_epoch=number_of_epochs, verbose=1, callbacks=None, validation_split=0.11)

# create two generators for training and validation
trainGen = load_data.genBatch()
valGen = load_data.genBatch()
evalGen = load_data.genBatch()

history = model.fit_generator(trainGen,
                              samples_per_epoch=number_of_samples_per_epoch,
                              nb_epoch=number_of_epochs,
                              validation_data=valGen,
                              nb_val_samples=number_of_validation_samples,
                              verbose=1)


# score = model.evaluate_generator(evalGen, 1000, max_q_size=10)
# print('Evaluation accuracy: %.2f%%', score[1]*100)

# save the weights
model.save_weights(weights_path+'/weights.h5')
#save the model with weights
model.save(weights_path+'/model_weights.h5')

# plots
# summarize history for loss
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='upper left')
plt.savefig('speed prediction loss plot')
print("Saved loss plot to disk")
plt.close()
