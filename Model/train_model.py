import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.regularizers import l2

from Helpers import helpers, given
from Model import model as m1

# Initialize fixed parameters
PATCH_SIZE = 16
STRIDE = PATCH_SIZE
WINDOW_SIZE = 32
PADDING = (WINDOW_SIZE - PATCH_SIZE) // 2


FILEPATH_SAVE_WEIGHTS = 'weights.h5'


# PUT YOUR DATAPATHS HERE
DATAPATH_TRAINING = "Datapath"
DATAPATH_TESTING = "Datapath"

NUM_TRAIN_IMAGES = 100
NUM_TEST_IMAGES = 50

PATCHES_PER_IMG = 625 #Somewhat random number

imgs, gt_imgs = helpers.load_training_data(DATAPATH_TRAINING, NUM_TRAIN_IMAGES)
imgs_zero = helpers.zero_mean(imgs, 400, std=False)

# Setting some callbacks. A callback is something that will be
# activated in the model.fit if certain criterias are fulfilled.
lr_callback = ReduceLROnPlateau(monitor='acc',        # Reduces learning rate if accuracy does not improve for 4 epochs
                                factor=0.5,
                                patience=4,
                                verbose=1,
                                mode='auto',
                                epsilon=0.0001,
                                cooldown=0,
                                min_lr=0)

stop_callback = EarlyStopping(monitor='acc',          # Stops the training if accuracy does not improve for 9 epochs
                              min_delta=0.0001,
                              patience=9,
                              verbose=1,
                              mode='auto')


def train():
    model = m1.create_model()
    adam = Adam(lr=0.001)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    # Create patches of size window_size from the training data
    x_train, y_train = helpers.create_random_windows_of_training_data(imgs_zero, gt_imgs,
                                                                      PATCHES_PER_IMG,
                                                                      WINDOW_SIZE)
    print("\nTraining data with size: ", x_train.shape)
    print("The size of the y-data: ", y_train.shape, "\n")
    try:
        model.fit(x_train, y_train,
                  epochs=200,
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

def predict_testdata(model):
    # Fix the test data
    imgs = helpers.load_test_data(DATAPATH_TESTING, NUM_TEST_IMAGES)
    x_test = helpers.zero_mean(imgs, 608, std=False)
    print("Creating patches...")
    img_patches = helpers.create_patches_test_data(x_test, PATCH_SIZE, STRIDE, PADDING)

    filename = 'submission.csv'

    # Run prediction
    print("Predicting...")
    pred = model.predict(img_patches)
    prediction = helpers.make_pred(pred)
    print("Prediction ready. Creating file...")

    submission = helpers.create_submission_format()
    given.create_csv_submission(submission, prediction, filename)


model = train()
predict_testdata(model)