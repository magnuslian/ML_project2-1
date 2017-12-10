import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from keras.models import Sequential, load_model, save_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam

from Helpers import helpers
from Given import given, proHelpers


PATCH_SIZE = 16
WINDOW_SIZE = 24
PADDING = (WINDOW_SIZE - PATCH_SIZE) // 2
POOL_SIZE = (2,2)
FOREGROUND_THRESHOLD = 0.25
NEURONS = 32


FILEPATH_SAVE_WEIGHTS = 'weights_win24.h5'


DATAPATH_TRAINING = "C:\\Users\\magnu\\Documents\\NTNU\\3 (Utveksling EPFL)\\Machine Learning\\Prosjekt2\\Data\\training\\"
DATAPATH_TESTING = "C:\\Users\\magnu\\Documents\\NTNU\\3 (Utveksling EPFL)\\Machine Learning\\Prosjekt2\\Data\\test_set_images\\"

NUM_TRAIN_IMAGES = 100
NUM_TEST_IMAGES = 50

PATCHES_PER_IMG = 625

#Load training data
imgs, gt_imgs = helpers.load_training_data(DATAPATH_TRAINING, NUM_TRAIN_IMAGES)

#Normalizing the training data
imgs_normal = helpers.normalize(imgs, 400, std=False)

#Setting some callbacks
lr_callback = ReduceLROnPlateau(monitor='acc',
                                factor=0.5,
                                patience=4,
                                verbose=1,
                                mode='auto',
                                epsilon=0.0001,
                                cooldown=0,
                                min_lr=0)

stop_callback = EarlyStopping(monitor='acc',
                              min_delta=0.0001,
                              patience=9,
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

    # Create patches of size window_size from the training data
    x_train, y_train = helpers.create_random_patches_of_training_data(imgs_normal, gt_imgs,
                                                                      PATCHES_PER_IMG,
                                                                      FOREGROUND_THRESHOLD,
                                                                      WINDOW_SIZE)
    print("\nTraining data with size: ", x_train.shape)
    print("The size of the y-data: ", y_train.shape, "\n")
    try:
        model.fit(x_train, y_train,
                  epochs=200,
                  batch_size=125,
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