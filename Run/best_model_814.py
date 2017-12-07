import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


from Helpers import helpers
from Given import given

DATAPATH_TRAINING = "C:\\Users\\magnu\\Documents\\NTNU\\3 (Utveksling EPFL)\\Machine Learning\\Prosjekt2\\Data\\training\\"
DATAPATH_TESTING = "C:\\Users\\magnu\\Documents\\NTNU\\3 (Utveksling EPFL)\\Machine Learning\\Prosjekt2\\Data\\test_set_images\\"

NUM_TRAIN_IMAGES = 100
NUM_TEST_IMAGES = 50

PATCH_SIZE = 16
#WINDOW_SIZE = 16
#PADDING = (WINDOW_SIZE - PATCH_SIZE) // 2
CLASSES = 2
POOL_SIZE = (2,2)
INPUT_SHAPE = (16,16,3)
BATCH_SIZE = 32
EPOCHS = [125, 150, 175]
FOREGROUND_THRESHOLD = 0.25

#NUM_OF_IMGS_CREATED_PER_IMGS = 8

weights_root = "C:\\Users\\magnu\\Documents\\NTNU\\3 (Utveksling EPFL)\\Machine Learning\\Prosjekt2\\ML_project2-1\\Saved_Weights\\"

# Load the training data and create patches of size PATCH_SIZE
imgs, gt_imgs = helpers.load_training_data(DATAPATH_TRAINING, NUM_TRAIN_IMAGES)
x_train, y_train = helpers.create_patches(imgs, FOREGROUND_THRESHOLD, PATCH_SIZE, gt_imgs=gt_imgs) #(62500,16,16,3), (62500,2)

#Normalizing train data
x_train = helpers.normalize(x_train, 16, std=False)

#
#       Increase training data
#
#x_ptrain, y_ptrain = helpers.increase_training_set(x_train,y_train,NUM_OF_IMGS_CREATED_PER_IMGS,WINDOW_SIZE)

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


for EPOCH in EPOCHS:
    model = Sequential()

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
    model.add(Dense(CLASSES, activation='softmax'))

    print("\nCompiling the model...")
    adam = Adam(lr=0.001, beta_1=0.7)
    model.compile(adam, 'categorical_crossentropy', metrics=['accuracy'])

    print("Training the model with epoch = ", EPOCH)
    model.fit(x_train,y_train,
              epochs=EPOCH,
              batch_size=BATCH_SIZE,
              verbose=2,
              callbacks=[lr_callback,stop_callback])

    #model.summary()

    filepath_weights = weights_root + str(EPOCH) + "EpochsWeights.h5"
    model.save_weights(filepath_weights, overwrite=False)
    print("\nWeights are saved.")

    # Load the test data to be used for delivery and create patches of size PATCH_SIZE
    imgs_test = helpers.load_test_data(DATAPATH_TESTING, NUM_TEST_IMAGES)
    x_test = helpers.create_patches(imgs_test,FOREGROUND_THRESHOLD,PATCH_SIZE) #(72200, 16, 16, 3)

    # Normalizing the test data
    x_test = helpers.normalize(x_test, 16, std=False)

    print("Predicting...")
    pred = model.predict(x_test)
    del model
    pred1 = helpers.make_pred(pred)
    submission = helpers.create_submission_format()
    filename = str(EPOCH) + 'Epochs.csv'
    given.create_csv_submission(submission, pred1, filename)
    print("The submission file was made, you can find it in the Run directory after the whole process is done.")
