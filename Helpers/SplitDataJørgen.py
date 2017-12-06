import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam
from keras.models import Sequential
from keras.layers import Dense, LeakyReLU
from keras.regularizers import l2

import matplotlib.pyplot as plt

from Helpers import helpers


DATAPATH_TRAINING = "C:\\Users\\magnu\\Documents\\NTNU\\3 (Utveksling EPFL)\\Machine Learning\\Prosjekt2\\Data\\training\\"
NUM_TRAIN_IMAGES = 20


#Initialize hyperparameters to be tested
NUMBER_OF_NEURONS = [16, 32, 64, 128, 256]
REGS = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
LEAKY_RELU = [True, False]
ALPHAS = [0.1,0.2,0.3,0.4]

#Load training data
imgs, gt_imgs = helpers.load_training_data(DATAPATH_TRAINING, NUM_TRAIN_IMAGES)

x_train, y_train = helpers.create_patches(imgs,0.25,16,gt_imgs=gt_imgs)

#Normalizing the training data
x_train = helpers.normalize(x_train,16,std=False)


def create_model(neuron, reg, leakyRelu, alpha):
    model = Sequential()

    if leakyRelu:
        model.add(Conv2D(32, (3, 3), input_shape=(16, 16, 3)))
        model.add(LeakyReLU(alpha=alpha))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3)))
        model.add(LeakyReLU(alpha=alpha))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(neuron, kernel_regularizer=l2(reg)))
        model.add(LeakyReLU(alpha=alpha))
        model.add(Dropout(0.5))

        model.add(Dense(2, activation='softmax', kernel_regularizer=l2(reg)))
    else:
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(16, 16, 3)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(neuron, activation='relu', kernel_regularizer=l2(reg)))
        model.add(Dropout(0.5))

        model.add(Dense(2, activation='softmax', kernel_regularizer=l2(reg)))

    adam = Adam(lr= 0.001)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def find_best_hp(neurons, regs, leakyRelu, alphas):
    #Initialize variables for hyper_params. Can extend with more hyper_params.
    best_loss = 100
    best_neuron = -1
    best_reg = -1
    best_leaky = True
    best_alpha = 0
    #best_dropout = -1
    #best_epoch = -1

    for leaky in leakyRelu:
        for reg in regs:
            for alpha in alphas:
                for neuron in neurons:
                    print("\nLEAKY?: ", leaky,
                          "\nREG: ", reg,
                          "\nALPHA:", alpha,
                          "\nNEURON: ", neuron)

                    model = create_model(neuron, reg, leaky, alpha)
                    history = model.fit(x_train, y_train, epochs=5, batch_size=32, verbose=2, validation_split = 0.2)

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
                    plt.show(block=False)
                    """

                    validation_losses = history.history['val_loss']
                    val_loss = min(validation_losses)
                    #epoch = (np.where(validation_losses == val_loss))
                    if (val_loss < best_loss):
                        print("NEW LEADER WITH VALIDATION LOSS OF", val_loss)
                        best_loss = val_loss
                        best_reg = reg
                        best_leaky = leaky
                        best_alpha = alpha
                        best_neuron = neuron
                        #best_dropout = dropout
                        #best_epoch = epoch
    return best_loss, best_neuron, best_leaky, best_reg, best_alpha


#Search for hyper_params with split_data
best_loss, best_neuron, best_leaky, best_reg, best_alpha = find_best_hp(NUMBER_OF_NEURONS, REGS, LEAKY_RELU, ALPHAS)

print("BEST LOSS: ", best_loss,
      "\nLEAKY?: ", best_leaky,
      "\nBEST NEURON: ", best_neuron,
      "\nBEST REG: ", best_reg,
      "\nBEST ALPHA:", best_alpha)