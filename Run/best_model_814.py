import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from skimage.exposure import rescale_intensity

from Helpers import helpers
from Given import given

DATAPATH_TRAINING = "C:\\Users\\magnu\\Documents\\NTNU\\3 (Utveksling EPFL)\\Machine Learning\\Prosjekt2\\Data\\training\\"
DATAPATH_TESTING = "C:\\Users\\magnu\\Documents\\NTNU\\3 (Utveksling EPFL)\\Machine Learning\\Prosjekt2\\Data\\test_set_images\\"

NUM_TRAIN_IMAGES = 100
NUM_TEST_IMAGES = 50

PATCH_SIZE = 16
WINDOW_SIZE = 16
PADDING = (WINDOW_SIZE - PATCH_SIZE) // 2
CLASSES = 2
POOL_SIZE = (2,2)
INPUT_SHAPE = (16,16,3)
BATCH_SIZE = 125
EPOCHS = 100
FOREGROUND_THRESHOLD = 0.25

NUM_OF_IMGS_CREATED_PER_IMGS = 8
SUBWINDOW_SIZE = WINDOW_SIZE - 2*PADDING

FILEPATH_SAVE_WEIGHTS = "C:\\Users\\magnu\\Documents\\NTNU\\3 (Utveksling EPFL)\\Machine Learning\\Prosjekt2\\ML_project2-1\\Saved_Weights\\weights_BM.h5"


# Load the training data and create patches of size PATCH_SIZE
imgs, gt_imgs = helpers.load_training_data(DATAPATH_TRAINING, NUM_TRAIN_IMAGES)
x_train, y_train = helpers.create_patches(imgs, FOREGROUND_THRESHOLD, PATCH_SIZE, gt_imgs=gt_imgs) #(62500,16,16,3), (62500,2)

#Normalizing train data
x_train = helpers.normalize(x_train, 16, std=False)

# Building a bigger data set by manipulating the training data
x_ptrain = np.empty((x_train.shape[0] * NUM_OF_IMGS_CREATED_PER_IMGS, WINDOW_SIZE, WINDOW_SIZE, 3))
y_ptrain = np.empty((y_train.shape[0] * NUM_OF_IMGS_CREATED_PER_IMGS, 2))

for iter in range (0,NUM_OF_IMGS_CREATED_PER_IMGS):
    # Create e.g 8 versions of each patch
    for patch in range(x_train.shape[0]):
        subimage_x = x_train[patch]
        subimage_y = y_train[patch]

        # IMAGE AUGMENTATION
        """
            Assuming that we want to create 8 new patches per patch, 
            we have a 38.6% chance of getting 8 unique patches. 
        """

        # Contrast stretching
        if np.random.randint(2) == 0:
            subimage_x = rescale_intensity(subimage_x)

        # Random flip vertically
        if np.random.randint(2) == 0:
            subimage_x = np.flipud(subimage_x)

        # Random flip horizontally
        if np.random.randint(2) == 0:
            subimage_x = np.fliplr(subimage_x)

        # Random rotation in steps of 90°
        num_rot = np.random.randint(4)
        subimage_x = np.rot90(subimage_x, num_rot)

        # subimage_y has shape (16,16)
        # subimage_y = given.value_to_class(np.mean(subimage_x), FOREGROUND_THRESHOLD)
        # subimage_y has shape [1,0] or [0,1]
        x_ptrain[patch*NUM_OF_IMGS_CREATED_PER_IMGS + iter] = subimage_x
        y_ptrain[patch*NUM_OF_IMGS_CREATED_PER_IMGS + iter] = subimage_y
    print("Done creating version ", iter+1, " of the patches")

#Setting some callbacks
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
# input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
# this applies 32 convolution filters of size 3x3 each.
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=INPUT_SHAPE))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=POOL_SIZE))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=POOL_SIZE))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

print("Compiling the model...")
adam = Adam(lr=0.001, beta_1=0.7)
model.compile(adam,'categorical_crossentropy', metrics=['accuracy'])

print("Training the model...")
model.fit(x_ptrain,y_ptrain,
          epochs=EPOCHS,
          batch_size=BATCH_SIZE,
          verbose=2,
          callbacks=[lr_callback,stop_callback])

model.summary()

model.save_weights(FILEPATH_SAVE_WEIGHTS, overwrite=False)
print("Weights are saved.")

# Load the test data to be used for delivery and create patches of size PATCH_SIZE
imgs_test = helpers.load_test_data(DATAPATH_TESTING, NUM_TEST_IMAGES)
x_test = helpers.create_patches(imgs_test,FOREGROUND_THRESHOLD,PATCH_SIZE) #(72200, 16, 16, 3)

# Normalizing the test data
x_test = helpers.normalize(x_test, 16, std=False)

print("Predicting...")
pred = model.predict(x_test)
pred1 = helpers.make_pred(pred)
submission = helpers.create_submission_format()
given.create_csv_submission(submission, pred1, "epoch=100med8gangerTrening.csv")