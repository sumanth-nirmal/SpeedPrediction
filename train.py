number_of_epochs = 8
number_of_samples_per_epoch = 20032
number_of_validation_samples = 6400
learning_rate = 1e-4

model.compile(optimizer=Adam(learning_rate), loss="mse", )

# create two generators for training and validation
trainGen = processData.genBatch()
valGen = processData.genBatch()
evalGen = processData.genBatch()

history = model.fit_generator(trainGen,
                              samples_per_epoch=number_of_samples_per_epoch,
                              nb_epoch=number_of_epochs,
                              validation_data=valGen,
                              nb_val_samples=number_of_validation_samples,
                              verbose=1)

# score = model.evaluate_generator(evalGen, 1000, max_q_size=10)
# print('Test score:', score[0])
# print('Test accuracy:', score[1])

# save the weights
model.save_weights('weights.h5')

#save the model with weights
model.save('model.h5')
