import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from keras.models import Sequential, load_model, save_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam

from Helpers import helpers



WINDOW_SIZES = [16,20,24]
FOREGROUND_THRESHOLD = 0.25
PATCHES_PER_IMG = 625


DATAPATH_TRAINING = "C:\\Users\\magnu\\Documents\\NTNU\\3 (Utveksling EPFL)\\Machine Learning\\Prosjekt2\\Data\\training\\"
DATAPATH_TESTING = "C:\\Users\\magnu\\Documents\\NTNU\\3 (Utveksling EPFL)\\Machine Learning\\Prosjekt2\\Data\\test_set_images\\"

NUM_TRAIN_IMAGES = 100
NUM_TEST_IMAGES = 50



#Load training data
imgs, gt_imgs = helpers.load_training_data(DATAPATH_TRAINING, NUM_TRAIN_IMAGES)

#Normalizing the training data
imgs = helpers.normalize(imgs, 400, std=False)

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

def create_model(i, window):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(window, window, 3), padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    if i == 1:
        model.add(Conv2D(32, (3, 3), activation='relu',padding='same'))
    if i == 2:
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    if i == 3:
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    if i == 1:
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    if i == 2:
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    if i == 3:
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
    return model


def optimize(imgs, gt_imgs, window_size):
    best_loss = 100
    best_model = -1
    best_window = -1

    for i in range(4):
        for window in window_size:


            # Create patches of size window_size from the training data
            x_train, y_train = helpers.create_random_patches_of_training_data(imgs, gt_imgs,
                                                                              PATCHES_PER_IMG,
                                                                              FOREGROUND_THRESHOLD,
                                                                              window)

            model = create_model(i, window)

            print("\nTraining data with size: ", x_train.shape)
            print("The size of the y-data: ", y_train.shape, "\n")
            history = model.fit(x_train, y_train,
                                epochs=15,
                                batch_size=32,
                                verbose=2,
                                validation_split=0.2,
                                callbacks=[lr_callback, stop_callback])
            print("\nDone training.")

            validation_losses = history.history['val_loss']
            val_loss = min(validation_losses)
            print("\nThis models min val_loss\n: ", val_loss)
            if (val_loss < best_loss):
                print("NEW LEADER WITH VALIDATION LOSS OF", val_loss, "\n")
                best_loss = val_loss
                best_model = i
                best_window = window
    return best_loss, best_model, best_window


#Search for hyper_params with split_data
best_loss, best_model, best_window  = optimize(imgs, gt_imgs, WINDOW_SIZES)

print("\nBEST LOSS: ", best_loss,
      "\nFROM MODEL: ", best_model,
      "\nBEST WINDOW: ", best_window)