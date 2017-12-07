import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np

from keras.models import Sequential, load_model, save_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam

from Helpers import helpers
from Given import given, proHelpers


PATCH_SIZE = 16
WINDOW_SIZE = 20
PADDING = (WINDOW_SIZE - PATCH_SIZE) // 2
CLASSES = 2
POOL_SIZE = (2,2)
FOREGROUND_THRESHOLD = 0.25


FILEPATH_SAVE_WEIGHTS = 2 #Put a string here


DATAPATH_TRAINING = "training/"
DATAPATH_TESTING = "test_set_images/"

NUM_TRAIN_IMAGES = 20
NUM_TEST_IMAGES = 50



#Load training data
x_train, y_train = helpers.load_training_data(DATAPATH_TRAINING, NUM_TRAIN_IMAGES)

x_train, y_train = helpers.create_patches(x_train, FOREGROUND_THRESHOLD, WINDOW_SIZE, y_train)

#Normalizing the training data
x_train = helpers.normalize(x_train, WINDOW_SIZE, std=False)



def create_model():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(WINDOW_SIZE, WINDOW_SIZE, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    adam = Adam(lr= 0.001)
    print("Compiling model......")
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

print("CREATING MODEL......")
model = create_model()

print("FITTING......")
history = model.fit(x_train, y_train, epochs=1, batch_size=64, verbose=2, validation_split = 0.2)
model.summary()
# summarize history for accuracy

#validation_losses = history.history['val_loss']
#val_loss = min(validation_losses)
#epoch = (np.where(validation_losses == val_loss))
#print(val_loss)

#model.save_weights(FILEPATH_SAVE_WEIGHTS, overwrite=False)
#print("Weights are saved.")
"""
# Load the test data to be used for delivery and create patches of size PATCH_SIZE
imgs_test = helpers.load_test_data(DATAPATH_TESTING, NUM_TEST_IMAGES)
x_test = helpers.create_patches(imgs_test,FOREGROUND_THRESHOLD,PATCH_SIZE)

# Normalizing the test data
x_test = helpers.normalize(x_test, 16, std=False)

print("Predicting...")
pred = model.predict(x_test)
pred1 = helpers.make_pred(pred)
submission = helpers.create_submission_format()
given.create_csv_submission(submission, pred1, "epoch=100med8gangerTrening.csv")"""


submission_filename = 'submission.csv'
image_filenames = []
root = "test_set_images/"
for i in range(1, 51):
    image_filename = root + '/test_' + str(i) + '/test_' + str(i) + '.png'
    image_filenames.append(image_filename)

proHelpers.generate_submission(model, submission_filename, *image_filenames)