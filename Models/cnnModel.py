import numpy as np
import keras as k
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from keras.models import Sequential, load_model, save_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, History

from skimage import exposure

from Helpers import helpers
from Given import given

PATCH_SIZE = 16
WINDOW_SIZE = 16
PADDING = (WINDOW_SIZE - PATCH_SIZE) // 2
CLASSES = 2
POOL_SIZE = (2,2)
BATCH_SIZE = 32
EPOCHS = 200


def initialize():
    #reg_factor = 1e-6 #L2 reg-factor

    input_shape = (WINDOW_SIZE, WINDOW_SIZE, 3)

    model = Sequential()
    # input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
    # this applies 32 convolution filters of size 3x3 each.
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
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
    return model


def train(model, x_train, y_train, save_model=True):

    print("Training a data set of shape: ", x_train.shape)
    print("With shape of y: ", y_train.shape)

    # Normalize training data
    x_train = helpers.normalize(x_train, 400, std=False)

    NUM_OF_IMGS_CREATED_PER_IMGS = 8

    x_ptrain = np.empty((x_train.shape[0] * NUM_OF_IMGS_CREATED_PER_IMGS, WINDOW_SIZE, WINDOW_SIZE, 3))
    y_ptrain = np.empty((y_train.shape[0] * NUM_OF_IMGS_CREATED_PER_IMGS, 2))

    #Iterations
    for iter in range (1,NUM_OF_IMGS_CREATED_PER_IMGS+1):
        # Create e.g 8 versions of each image
        for pic in range(x_train.shape[0]):
            # Random image in the list
            ind = np.random.randint(0,x_train.shape[0])
            shape = x_train[ind] #.shape?

            # Random window from the image
            randomw = np.random.randint(0, shape[0] - WINDOW_SIZE)
            randomh = np.random.randint(0, shape[1] - WINDOW_SIZE)
            subimage_x = x_train[ind][randomw:randomw + WINDOW_SIZE, randomh:randomh + WINDOW_SIZE]
            subimage_y = y_train[ind][randomw:randomw + WINDOW_SIZE, randomh:randomh + WINDOW_SIZE]

            # Reflex
            subimage_x = np.lib.pad(subimage_x, ((PADDING, PADDING), (PADDING, PADDING), (0, 0)), 'reflect')
            subimage_y = np.lib.pad(subimage_y, ((PADDING, PADDING), (PADDING, PADDING)), 'reflect')


            # IMAGE AUGMENTATION

            # Contrast stretching
            if np.random.randint(2) == 0:
                p2, p98 = np.percentile(subimage_x,(2,98))                              #What is it supposed to be?
                subimage_x = exposure.rescale_intensity(subimage_x, in_range=(p2, p98)) #What is it supposed to be?

            # Random flip
            if np.random.randint(2) == 0:
                # Flip vertically
                subimage_x = np.flipud(subimage_x)
            if np.random.randint(2) == 0:
                # Flip horizontally
                subimage_x = np.fliplr(subimage_x)

            # Random rotation in steps of 90°
            num_rot = np.random.randint(4)
            subimage_x = np.rot90(subimage_x, num_rot)

            #38,5 % chance of 8 unique images created.
            #Why: 32 possible images to create.
            #The first time you will draw a unique image (32/32)
            #Next time, the chance is 31/32 etc.
            #(32*31*30*29*28*27*26*25)/(32^8) = 38,5%. This means that over 100 images,~60 of them will create duplicates
            #If adding another 50/50 test, we get 63,4%.
            #Should we rather have that?

            subimage_y = k.utils.to_categorical(subimage_y, num_classes=2) #should this be subimage_y?
            x_ptrain[pic*NUM_OF_IMGS_CREATED_PER_IMGS + iter] = subimage_x
            y_ptrain[pic*NUM_OF_IMGS_CREATED_PER_IMGS + iter] = subimage_y


    adam = Adam(lr=0.001, beta_1=0.7)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])

    # CALLBACKS
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

    history = History()


    model.fit(x_train, y_train,
              epochs=200,
              batch_size=BATCH_SIZE,
              verbose=0,
              callbacks=[lr_callback, stop_callback, history])

    print("Training done.")

    if save_model:
        save(model,'/model.h5',overwrite=False)
        print("Model saved.")

    return model


def prediction(model, x):
    # Make prediction and submission file
    test_image_patches = helpers.create_patches(x, 0.25, 16)

    print("Predicting...")
    pred = model.predict(test_image_patches)

    pred1 = helpers.make_pred(pred)
    submission = helpers.create_submission_format()
    given.create_csv_submission(submission, pred1, "epochs=200.csv")


def save(model, filepath, overwrite=True, include_opt=True):
    save_model(model,filepath,overwrite=overwrite,include_optimizer=include_opt)

def load(filepath, custom_obj=None, compileIt=True):
    load_model(filepath,custom_objects=custom_obj,compile=compileIt)