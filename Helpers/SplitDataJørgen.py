import numpy as np

from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l2

import matplotlib.pyplot as plt




from Helpers import helpers


DATAPATH_TRAINING = "training/"
NUM_TRAIN_IMAGES = 20


#Initialize hyperparameters to be tested
LEARNING_RATES = [0.001]
BETA_1 = [0.7]
NUMBER_OF_NEURONS = [16, 32, 64, 128, 256]
REGS = [1e-7, 1e-6, 1e-5]
DROPOUTS =[0.15, 0.2, 0.25, 0.3, 0.35]

#Load training data
x_train = helpers.extract_data(DATAPATH_TRAINING, NUM_TRAIN_IMAGES)
y_train = helpers.extract_labels(DATAPATH_TRAINING, NUM_TRAIN_IMAGES)


#Normalizing the training data
x_train = np.reshape(x_train, (NUM_TRAIN_IMAGES*160000, 3))
x_train = helpers.normalize(x_train,std=False)
x_train = np.reshape(x_train, (NUM_TRAIN_IMAGES*625,16,16,3))


def create_model(neuron, reg, dropout):
    model = Sequential()
    # input: 400x400 images with 3 channels -> (400, 400, 3) tensors.
    # this applies 32 convolution filters of size 3x3 each.
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(16, 16, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout))

    model.add(Flatten())
    model.add(Dense(neuron, activation='relu', kernel_regularizer=l2(reg)))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax', kernel_regularizer=l2(reg)))

    adam = Adam(lr= 0.001, beta_1=0.7,beta_2=0.999,epsilon=1e-8,decay=0)
    print("Compiling model......")
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def find_best_hp(neurons, regs, dropouts):
    #Initialize variables for hyper_params. Can extend with more hyper_params.
    best_loss = 100
    best_neuron = -1
    best_dropout = -1
    best_reg = -1
    best_epoch = -1
    #Grid search over hyper_param
    counter = 0
    best_model = 0
    for reg in regs:
        for dropout in dropouts:
            for neuron in neurons:
                counter += 1
                print("Neurons:", neuron)
                print("Fitting......")
                model = create_model(neuron, reg, dropout)
                history = model.fit(x_train, y_train, epochs=25, batch_size=64, verbose=2, validation_split = 0.2)
                """
                # summarize history for accuracy
                plt.plot(history.history['acc'])
                plt.plot(history.history['val_acc'])
                plt.title('model accuracy')
                plt.ylabel('accuracy')
                plt.xlabel('epoch')
                plt.legend(['train', 'test'], loc='upper left')
                plt.show()
                # summarize history for loss
                plt.plot(history.history['loss'])
                plt.plot(history.history['val_loss'])
                plt.title('model loss')
                plt.ylabel('loss')
                plt.xlabel('epoch')
                plt.legend(['train', 'test'], loc='upper left')
                plt.show()
                """
                validation_losses = history.history['val_loss']
                val_loss = min(validation_losses)
                epoch = (np.where(validation_losses == val_loss))
                if (val_loss < best_loss):
                    print("NEW LEADER WITH VALIDATION LOSS OF", val_loss)
                    best_loss = val_loss
                    best_reg = reg
                    best_dropout = dropout
                    best_neuron = neuron
                    best_epoch = epoch
    return best_loss, best_neuron, best_dropout, best_reg, best_epoch


#Search for hyper_params with split_data
best_loss, best_neuron, best_dropout, best_reg, best_epoch = find_best_hp(NUMBER_OF_NEURONS, REGS, DROPOUTS)

print("BEST NEURON: ", best_neuron,
      "BEST DROPOUT: ", best_dropout,
      "BEST EPOCH: ", best_epoch,
      "BEST LOSS:", best_loss)