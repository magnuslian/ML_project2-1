import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.regularizers import l2
from keras.optimizers import Adam

from Helpers import helpers

WINDOW_SIZE = 24
PATCHES_PER_IMG = 625

FILTER_SIZES = (5,5) #0.0001 best for (3,3), (5,5) best for 0.001
LEARNING_RATES = 0.001 #Notes: 0.1, 0.01 and 0.00001 doesn't learn (for either filter)
REGULIZERS = [1e-7,1e-6]
DROPOUTS = [0.1,0.15,0.2,0.25,0.3]
NEURONS = 512 #512 is the winner up to 2048.



DATAPATH_TRAINING = "C:\\Users\\magnu\\Documents\\NTNU\\3 (Utveksling EPFL)\\Machine Learning\\Prosjekt2\\Data\\training\\"
NUM_TRAIN_IMAGES = 20


#Load training data
imgs, gt_imgs = helpers.load_training_data(DATAPATH_TRAINING, NUM_TRAIN_IMAGES)

#Normalizing the training data
imgs = helpers.zero_mean(imgs, 400, std=False)

# Create patches of size window_size from the training data
x_train, y_train = helpers.create_random_patches_of_training_data(imgs, gt_imgs,
                                                                  PATCHES_PER_IMG,
                                                                  WINDOW_SIZE)


def create_model(filter, lr, neur, reg, drop):
    model = Sequential()

    model.add(Conv2D(32, filter, activation='relu', input_shape=(WINDOW_SIZE, WINDOW_SIZE, 3), padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(drop))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(drop))

    model.add(Flatten())
    model.add(Dense(neur, activation='relu', kernel_regularizer=l2(reg)))
    model.add(Dropout(drop*2))
    model.add(Dense(2, activation='softmax', kernel_regularizer=l2(reg)))

    adam = Adam(lr= lr)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def optimize(x_train, y_train):
    best_loss = 100
    best_drop = -1
    best_reg = -1
    for reg in REGULIZERS:
        for drop in DROPOUTS:
            model = create_model(FILTER_SIZES, LEARNING_RATES, NEURONS, reg, drop)

            #print("\nTraining data with size: ", x_train.shape)
            #print("The size of the y-data: ", y_train.shape, "\n")
            history = model.fit(x_train, y_train,
                                epochs=10,
                                batch_size=32,
                                verbose=2,
                                validation_split=0.2)

            validation_losses = history.history['val_loss']
            val_loss = min(validation_losses)
            print("\nDrop: ", drop)
            print("Reg: ", reg)
            if (val_loss < best_loss):
                print("\nNEW LEADER WITH VALIDATION LOSS OF", val_loss, "\n")
                best_loss = val_loss
                best_drop = drop
                best_reg = reg
            else:
                print("\nNot a new record :( Val loss: ", val_loss, "\n")
    return best_loss, best_drop, best_reg


#Search for hyper_params with split_data
best_loss, best_filter, best_reg  = optimize(x_train, y_train)

print("\nBEST LOSS: ", best_loss,
      "\nFROM DROP: ", best_filter,
      "\nWITH REG: ", best_reg)