import numpy as np
import keras as k

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.preprocessing import image

from Helpers import helpers
from Given import given

DATAPATH_TRAINING = "C:\\Users\\magnu\\Documents\\NTNU\\3 (Utveksling EPFL)\\Machine Learning\\Prosjekt2\\Data\\training\\"
DATAPATH_TESTING = "C:\\Users\\magnu\\Documents\\NTNU\\3 (Utveksling EPFL)\\Machine Learning\\Prosjekt2\\Data\\test_set_images\\"

NUM_TRAIN_IMAGES = 100
SPLIT_DATA_TRAIN = 100
SPLIT_DATA_TEST = 0

NUM_TEST_IMAGES = 50

imgs = helpers.extract_data(DATAPATH_TRAINING, NUM_TRAIN_IMAGES)
gt_imgs = helpers.extract_labels(DATAPATH_TRAINING, NUM_TRAIN_IMAGES)
#print(gt_imgs.shape)
#print(imgs.shape)

x_train = imgs[:(SPLIT_DATA_TRAIN*625),:,:,:]
y_train = gt_imgs[:(SPLIT_DATA_TRAIN*625)]
#print(x_train.shape)
#print(y_train.shape)

x_test = helpers.extract_test_data(DATAPATH_TESTING,NUM_TEST_IMAGES)

#x_test = imgs[:SPLIT_DATA_TEST*625]
#y_test = gt_imgs[:SPLIT_DATA_TEST*625]
#print(x_test.shape)
#print(y_test.shape)

#Normalizing the x_train
x_train = np.reshape(x_train, (SPLIT_DATA_TRAIN*160000, 3))
x_train = helpers.normalize(x_train,std=False)
x_train = np.reshape(x_train, (SPLIT_DATA_TRAIN*625,16,16,3))

#Normalizing the x_test
x_test = np.reshape(x_test, (NUM_TEST_IMAGES*369664, 3))
x_test = helpers.normalize(x_test,std=False)
x_test = np.reshape(x_test, (NUM_TEST_IMAGES*38*38,16,16,3))

model = Sequential()
# input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
# this applies 32 convolution filters of size 3x3 each.
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(16, 16, 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean squared error', optimizer=sgd)

model.fit(x_train, y_train, batch_size=32, epochs=2)
pred = model.predict(x_test)




pred1 = helpers.make_pred(pred)
submission = helpers.create_submission_format()
given.create_csv_submission(submission, pred1, "first_try_with_neural_nets.csv")