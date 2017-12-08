import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np

from keras.models import Sequential, load_model, save_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam

from Helpers import helpers
from Given import given, proHelpers


PATCH_SIZE = 16
WINDOW_SIZE = 20
PADDING = (WINDOW_SIZE - PATCH_SIZE) // 2
CLASSES = 2
POOL_SIZE = (2,2)
FOREGROUND_THRESHOLD = 0.25


FILEPATH_SAVE_WEIGHTS = 'weights_win20.h5'


DATAPATH_TRAINING = "C:\\Users\\magnu\\Documents\\NTNU\\3 (Utveksling EPFL)\\Machine Learning\\Prosjekt2\\Data\\training\\"
DATAPATH_TESTING = "C:\\Users\\magnu\\Documents\\NTNU\\3 (Utveksling EPFL)\\Machine Learning\\Prosjekt2\\Data\\test_set_images\\"

NUM_TRAIN_IMAGES = 10
NUM_TEST_IMAGES = 50

PATCHES_PER_IMG = round((400/WINDOW_SIZE)**2)



#Load training data
x_train, y_train = helpers.load_training_data(DATAPATH_TRAINING, NUM_TRAIN_IMAGES)

#Normalizing the training data
x_train = helpers.normalize(x_train, 400, std=False)

#Create patches of size window_size from the training data
x_train, y_train = helpers.create_random_patches_of_training_data(x_train,y_train, PATCHES_PER_IMG, FOREGROUND_THRESHOLD, WINDOW_SIZE)

lr_callback = ReduceLROnPlateau(monitor='acc',
                                        factor=0.5,
                                        patience=5,
                                        verbose=1,
                                        mode='auto',
                                        epsilon=0.0001,
                                        cooldown=0,
                                        min_lr=0)

stop_callback = EarlyStopping(monitor='acc',
                          min_delta=0.0001,
                          patience=11,
                          verbose=1,
                          mode='auto')


model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(WINDOW_SIZE, WINDOW_SIZE, 3)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

adam = Adam(lr= 0.001)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

print("\nTraining data with size: ", x_train.shape)
print("The size of the y-data: ", y_train.shape, "\n")
model.fit(x_train, y_train, epochs=1, batch_size=32, verbose=2, callbacks=[lr_callback,stop_callback])
print("\nDone training.\n")
#model.summary()

model.save_weights(FILEPATH_SAVE_WEIGHTS, overwrite=False)
print("Weights are saved.\n")

imgs = helpers.load_test_data(DATAPATH_TESTING, NUM_TEST_IMAGES)

x_test = helpers.normalize(imgs, 608, std=False)

pred = proHelpers.classify(model, x_test)
submission = helpers.create_submission_format()
given.create_csv_submission(submission,pred,"window20.csv")