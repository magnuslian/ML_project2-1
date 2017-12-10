import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np

from keras.models import Sequential, load_model, save_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from skimage.exposure import rescale_intensity

from Helpers import helpers
from Given import given, proHelpers


PATCH_SIZE = 16
WINDOW_SIZE = 24
PADDING = (WINDOW_SIZE - PATCH_SIZE) // 2
POOL_SIZE = (2,2)
NEURONS = 32
BATCH_SIZE = 125

FILEPATH_SAVE_WEIGHTS = 'weights_win24.h5'

# EDIT THESE THREE PATHS BEFORE RUNNING
DATAPATH_TRAINING = "C:\\Users\\magnu\\Documents\\NTNU\\3 (Utveksling EPFL)\\Machine Learning\\Prosjekt2\\Data\\training\\"
DATAPATH_TESTING = "C:\\Users\\magnu\\Documents\\NTNU\\3 (Utveksling EPFL)\\Machine Learning\\Prosjekt2\\Data\\test_set_images\\"

NUM_TRAIN_IMAGES = 100
NUM_TEST_IMAGES = 50

#Load training data
imgs, gt_imgs = helpers.load_training_data(DATAPATH_TRAINING, NUM_TRAIN_IMAGES)

#Normalizing the training data
imgs_normal = helpers.normalize(imgs, 400, std=False)

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

def create_model():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(WINDOW_SIZE, WINDOW_SIZE, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=POOL_SIZE))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=POOL_SIZE))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(NEURONS, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    adam = Adam(lr= 0.001)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train():
    model = create_model()

    def generate_minibatch():
        # Loop for ever
        while True:
            # Create one batch of training data
            x_batch = np.empty((BATCH_SIZE, WINDOW_SIZE, WINDOW_SIZE, 3))
            y_batch = np.empty((BATCH_SIZE, 2))

            for iter in range(0, BATCH_SIZE):
                #Choose random image
                index = np.random.randint(0,imgs_normal.shape[0])

                width = imgs_normal[index].shape[0]
                height = imgs_normal[index].shape[1]

                # Random window from the image
                randomw = np.random.randint(0, width - WINDOW_SIZE + 1)  # +1 includes one below (400 - 20 + 1 =) 381
                # randomw is a random number which decides the starting column (leftmost)
                randomh = np.random.randint(0, height - WINDOW_SIZE + 1)
                # randomh is a random number which decides the starting row (uppermost)
                window_x = imgs_normal[index][randomw:randomw + WINDOW_SIZE, randomh:randomh + WINDOW_SIZE]
                window_y = imgs_normal[index][randomw:randomw + WINDOW_SIZE, randomh:randomh + WINDOW_SIZE]

                # subimage_y is independent of the image augmentation of subimage_x
                window_y = given.value_to_class(np.mean(window_y))

                # IMAGE AUGMENTATION

                # Contrast stretching
                if np.random.randint(2) == 0:
                    subimage_x = rescale_intensity(window_x)

                # Random flip vertically
                if np.random.randint(2) == 0:
                    subimage_x = np.flipud(window_x)

                # Random flip horizontally
                if np.random.randint(2) == 0:
                    subimage_x = np.fliplr(window_x)

                # Random rotation in steps of 90°
                num_rot = np.random.randint(4)
                subimage_x = np.rot90(window_x, num_rot)



                x_batch[iter] = window_x
                y_batch[iter] = window_y
            yield (x_batch,y_batch)


    print("\nTraining data with size: ", imgs_normal.shape)
    print("The size of the y-data: ", gt_imgs.shape, "\n")
    try:
        model.fit_generator(generate_minibatch(),
                            steps_per_epoch=500, # 500 * batch_size = 62500 which should be representative for the data set
                            epochs=200,
                            verbose=2,
                            callbacks=[lr_callback,stop_callback])
    except KeyboardInterrupt:
        print("Keyboard interrupt")
        pass

    print("\nDone training.\n")

    model.summary()

    try:
        model.save_weights(FILEPATH_SAVE_WEIGHTS, overwrite=False)
        print("Weights are saved.\n")
    except Exception:
        print("Filename not found. Saving default filename.")
        filepath = 'weights.h5'
        model.save_weights(filepath, overwrite=False)
        print("Weights are saved.\n")
    return model

def predicting_that_shit(model, filename, test_patches):
    # Run prediction
    print("Predicting...")
    Z = model.predict(test_patches)
    Zi = helpers.make_pred(Z)
    print("Prediction ready. Creating file...")

    submission = helpers.create_submission_format()
    given.create_csv_submission(submission, Zi, filename)

#Fix the test data
imgs = helpers.load_test_data(DATAPATH_TESTING, NUM_TEST_IMAGES)
x_test = helpers.normalize(imgs, 608, std=False)
print("Creating patches...")
img_patches = proHelpers.create_patches(x_test, 16, 16, PADDING)

filename = 'window24.csv'

model = train()
predicting_that_shit(model,filename,img_patches)