import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from keras.models import Sequential, load_model, save_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.regularizers import l2

from Helpers import helpers
from Given import given


PATCH_SIZE = 16
WINDOW_SIZE = 32
PADDING = (WINDOW_SIZE - PATCH_SIZE) // 2
POOL_SIZE = (2,2)
NEURONS = 256
FIRST_FILTER = (5,5)
DROPOUT = 0.25
REGULIZER = 1e-7


FILEPATH_SAVE_WEIGHTS = 'weights_best.h5'


DATAPATH_TRAINING = "C:\\Users\\magnu\\Documents\\NTNU\\3 (Utveksling EPFL)\\Machine Learning\\Prosjekt2\\Data\\training\\"
DATAPATH_TESTING = "C:\\Users\\magnu\\Documents\\NTNU\\3 (Utveksling EPFL)\\Machine Learning\\Prosjekt2\\Data\\test_set_images\\"

NUM_TRAIN_IMAGES = 100
NUM_TEST_IMAGES = 50

PATCHES_PER_IMG = 625

#Load training data
imgs, gt_imgs = helpers.load_training_data(DATAPATH_TRAINING, NUM_TRAIN_IMAGES)

#Zero-center the training data
imgs_zero = helpers.zero_mean(imgs, 400, std=False) #Had normalize before

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

    model.add(Conv2D(32, FIRST_FILTER, activation='relu', padding='same', input_shape=(WINDOW_SIZE, WINDOW_SIZE, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=POOL_SIZE))
    model.add(Dropout(DROPOUT))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=POOL_SIZE))
    model.add(Dropout(DROPOUT))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=POOL_SIZE))
    model.add(Dropout(DROPOUT))

    model.add(Flatten())
    model.add(Dense(NEURONS, activation='relu', kernel_regularizer=l2(REGULIZER)))
    model.add(Dropout(DROPOUT*2))
    model.add(Dense(2, activation='softmax', kernel_regularizer=l2(REGULIZER)))

    adam = Adam(lr= 0.001)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train():
    model = create_model()

    # Create patches of size window_size from the training data
    x_train, y_train = helpers.create_random_patches_of_training_data(imgs_zero, gt_imgs,
                                                                      PATCHES_PER_IMG,
                                                                      WINDOW_SIZE)
    print("\nTraining data with size: ", x_train.shape)
    print("The size of the y-data: ", y_train.shape, "\n")
    try:
        model.fit(x_train, y_train,
                  epochs=500,
                  batch_size=125,
                  verbose=2,
                  callbacks=[lr_callback,stop_callback])
    except KeyboardInterrupt: #Do not ruin the model if user interrupts
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

def predicting_that_shit(model):
    # Fix the test data
    imgs = helpers.load_test_data(DATAPATH_TESTING, NUM_TEST_IMAGES)
    x_test = helpers.zero_mean(imgs, 608, std=False)
    print("Creating patches...")
    img_patches = helpers.create_patches_test_data(x_test, 16, 16, PADDING)

    filename = 'window32.csv'

    # Run prediction
    print("Predicting...")
    Z = model.predict(img_patches)
    Zi = helpers.make_pred(Z)
    print("Prediction ready. Creating file...")

    submission = helpers.create_submission_format()
    given.create_csv_submission(submission, Zi, filename)



model = train()
predicting_that_shit(model)