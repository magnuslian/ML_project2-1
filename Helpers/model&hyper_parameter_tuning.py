import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.regularizers import l2
from keras.optimizers import Adam

from Helpers import helpers

# Initialize fixed parameters
WINDOW_SIZE = 32
PATCHES_PER_IMG = 625
NEURONS = 128
FILTER_SIZE = (5,5)
REGULARIZER = 1e-7
EPOCHS = 50

# Initialize parameters to be tested. This works as an example of possible hyper parameters to test
LEARNING_RATES = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
DROPOUTS = [0.1,0.15,0.2,0.25,0.3]


DATAPATH_TRAINING = "Datapath" # INSERT CORRECT DATAPATH HERE
NUM_TRAIN_IMAGES = 20


imgs, gt_imgs = helpers.load_training_data(DATAPATH_TRAINING, NUM_TRAIN_IMAGES)
imgs = helpers.zero_mean(imgs, 400, std=False)
x_train, y_train = helpers.create_random_windows_of_training_data(imgs, gt_imgs,
                                                                  PATCHES_PER_IMG,
                                                                  WINDOW_SIZE)

def create_model(lr, drop):
    """Creating the model.

    The input parameters depend on what parameters we want to evaluate.
    Learning rate and dropout is provided as an example.
    """
    model = Sequential()

    model.add(Conv2D(32, FILTER_SIZE, activation='relu', input_shape=(WINDOW_SIZE, WINDOW_SIZE, 3), padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(drop))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(drop))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(drop))

    model.add(Flatten())
    model.add(Dense(NEURONS, activation='relu', kernel_regularizer=l2(REGULARIZER)))
    model.add(Dropout(drop*2))
    model.add(Dense(2, activation='softmax', kernel_regularizer=l2(REGULARIZER)))

    adam = Adam(lr= lr)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train_and_evaluate(x_train, y_train):
    """Finds the optimal combination of the given hyper parameters

    Using keras.fit with a validation set of 20% of the training data.
    :param x_train: Training images
    :param y_train: Corresponding ground truth labels of training images
    :return: The best combination of hyper parameters with the corresponding validation loss
    """
    best_loss = 100
    best_drop = -1
    best_learning_rate = -1
    for learning_rate in LEARNING_RATES:
        for drop in DROPOUTS:
            print("\nLR: ", learning_rate)
            print("Drop: ", drop)

            model = create_model(learning_rate, drop)   #Create the model with the given parameters

            #Fit the model. We use validation split of 0.2.
            history = model.fit(x_train, y_train,
                                epochs=EPOCHS,
                                batch_size=32,
                                verbose=2,
                                validation_split=0.2)

            validation_losses = history.history['val_loss'] #Extracting the validation loss per epoch
            val_loss = min(validation_losses)               #Finding the minimum validation loss

            #Updating the best validation loss, and the best hyper parameters
            if (val_loss < best_loss):
                print("\nNEW LEADER WITH VALIDATION LOSS OF", val_loss, "\n")
                best_loss = val_loss
                best_drop = drop
                best_learning_rate = learning_rate
            else:
                print("\nNot a new record 😞 Val loss: ", val_loss, "\n")

    return best_loss, best_drop, best_learning_rate

best_loss, best_filter, best_LR  = train_and_evaluate(x_train, y_train)

print("\nBEST LOSS: ", best_loss,
      "\nFROM DROP: ", best_filter,
      "\nWITH LEARNING RATE: ", best_LR)