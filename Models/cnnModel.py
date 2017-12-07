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
from Given import given, proHelpers

PATCH_SIZE = 16
WINDOW_SIZE = 32
PADDING = (WINDOW_SIZE - PATCH_SIZE) // 2
CLASSES = 2
POOL_SIZE = (2,2)
BATCH_SIZE = 200
EPOCHS = 200
FOREGROUND_THRESHOLD = 0.25

FILEPATH_SAVE_WEIGHTS = 2 #Put a string here


    def initialize(self):
        #reg_factor = 1e-6 #L2 reg-factor

        input_shape = (WINDOW_SIZE, WINDOW_SIZE, 3)

        self.model = Sequential()
        # input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
        # this applies 32 convolution filters of size 3x3 each.
        self.model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
        self.model.add(Conv2D(32, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=POOL_SIZE))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=POOL_SIZE))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(16, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(2, activation='softmax'))


    def train(self, x_train, y_train, save_weights=True):

        print("Training a data set of shape: ", x_train.shape)
        print("With shape of y: ", y_train.shape)

        # Normalize training data
        x_train = helpers.normalize(x_train, 400, std=False)

        NUM_OF_IMGS_CREATED_PER_IMGS = 200
        SUBWINDOW_SIZE = WINDOW_SIZE - 2*PADDING

        x_ptrain = np.empty((x_train.shape[0] * NUM_OF_IMGS_CREATED_PER_IMGS, WINDOW_SIZE, WINDOW_SIZE, 3))
        y_ptrain = np.empty((y_train.shape[0] * NUM_OF_IMGS_CREATED_PER_IMGS, 2))

        #Iterations
        for iter in range (0,NUM_OF_IMGS_CREATED_PER_IMGS):
            # Create e.g 8 versions of each image
            for pic in range(x_train.shape[0]):
                # Random image in the list
                ind = np.random.randint(0,x_train.shape[0])
                shape = x_train[ind].shape

                # Random window from the image
                randomw = np.random.randint(0, shape[0] - SUBWINDOW_SIZE)
                randomh = np.random.randint(0, shape[1] - SUBWINDOW_SIZE)
                subimage_x = x_train[ind][randomw:randomw + SUBWINDOW_SIZE, randomh:randomh + SUBWINDOW_SIZE]
                subimage_y = y_train[ind][randomw:randomw + SUBWINDOW_SIZE, randomh:randomh + SUBWINDOW_SIZE]

                # Reflex
                subimage_x = np.lib.pad(subimage_x, ((PADDING, PADDING), (PADDING, PADDING), (0, 0)), 'reflect')
                subimage_y = np.lib.pad(subimage_y, ((PADDING, PADDING), (PADDING, PADDING)), 'reflect')


                # IMAGE AUGMENTATION

                """
                # Contrast stretching
                if np.random.randint(2) == 0:
                    p2, p98 = np.percentile(subimage_y,(2,98))                              #What is it supposed to be?
                    subimage_y = exposure.rescale_intensity(subimage_y, in_range=(p2, p98)) #What is it supposed to be?
                """

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

                # subimage_y has shape (16,16)
                subimage_y = given.value_to_class(np.mean(subimage_y), FOREGROUND_THRESHOLD)
                # subimage_y has shape [1,0] or [0,1]
                x_ptrain[pic*NUM_OF_IMGS_CREATED_PER_IMGS + iter] = subimage_x
                y_ptrain[pic*NUM_OF_IMGS_CREATED_PER_IMGS + iter] = subimage_y


        adam = Adam(lr=0.001, beta_1=0.7)
        self.model.compile(loss='categorical_crossentropy',
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


        self.model.fit(x_ptrain, y_ptrain,
                  batch_size=BATCH_SIZE,
                  epochs=200,
                  verbose=2,
                  callbacks=[lr_callback,stop_callback,history])

        self.model.summary()

        print("Training done.")

        if save_weights:
            #try:
            self.model.save_weights(FILEPATH_SAVE_WEIGHTS, overwrite=False)
            print("Model saved.")

    def load(self, filename):
        self.model.load_weights(filename)

    def prediction(self, x):
        # Make prediction and submission file
        test_image_patches = helpers.create_patches(x, 0.25, 16)

        print("Predicting...")
        pred = self.model.predict(test_image_patches)
        """
        pred1 = helpers.make_pred(pred)
        submission = helpers.create_submission_format()
        given.create_csv_submission(submission, pred1, "new_model_setup.csv")"""


        pred1 = helpers.make_pred(pred)
        pred2 = np.reshape(pred1, (50, 38, 38))
        pred3 = helpers.filterh(pred2)
        pred4 = np.reshape(pred3, (72200, 1))
        submission = helpers.create_submission_format()
        given.create_csv_submission(submission, pred4, "cleaned .csv")


    #THE PROS
    def classify(self, X):
        """
        Classify an unseen set of samples.
        This method must be called after "train".
        Returns a list of predictions.
        """

        # Subdivide the images into blocks.
        # img_patches has shape (1444, 32, 32, 3)
        img_patches = proHelpers.create_patches(X, self.patch_size, 16, self.padding)

        # Run prediction
        Z = self.model.predict(img_patches)
        # Z has shape (1444, 2)

        Z = (Z[:, 0] < Z[:, 1]) * 1 # converts true to 1 and false to 0.
        # Z is an array of 1444

        # Regroup patches into images
        return proHelpers.group_patches(Z, X.shape[0]) #Shape of X is (1,608,608,3)