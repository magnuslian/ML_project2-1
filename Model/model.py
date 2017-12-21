"""Our model.

This is the model behind the 'weights_best.h5'.
The corresponding submission file is 'window32.csv'
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.regularizers import l2

# Initialize fixed parameters
WINDOW_SIZE = 32
POOL_SIZE = (2,2)
NEURONS = 128
FIRST_FILTER = (5,5)
DROPOUT = 0.25
REGULARIZER = 1e-7

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
    model.add(Dense(NEURONS, activation='relu', kernel_regularizer=l2(REGULARIZER)))
    model.add(Dropout(DROPOUT * 2))
    model.add(Dense(2, activation='softmax', kernel_regularizer=l2(REGULARIZER)))

    return model