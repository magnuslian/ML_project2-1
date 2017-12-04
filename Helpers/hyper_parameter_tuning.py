import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam
from keras.models import Sequential
from keras.layers import Dense

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score, GridSearchCV

from Helpers import helpers


DATAPATH_TRAINING = "C:\\Users\\magnu\\Documents\\NTNU\\3 (Utveksling EPFL)\\Machine Learning\\Prosjekt2\\Data\\training/"
NUM_TRAIN_IMAGES = 20
K_FOLDS = 4                                       #Initialize K-fold
LEARNING_RATES = 0.0011 #Initialize learning rates
#BETA_1 = [0.7,0.8,0.9,0.99]
NEURONS = [128, 256, 512, 1024, 2048, 4096]

#Load training data
x_train = helpers.extract_data(DATAPATH_TRAINING, NUM_TRAIN_IMAGES)
y_train = helpers.extract_labels(DATAPATH_TRAINING, NUM_TRAIN_IMAGES)


#Normalizing the training data
x_train = np.reshape(x_train, (NUM_TRAIN_IMAGES*160000, 3))
x_train = helpers.normalize(x_train,std=False)
x_train = np.reshape(x_train, (NUM_TRAIN_IMAGES*625,16,16,3))



def create_model(learning_rate, neuron, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0):
    model = Sequential()
    # input: 400x400 images with 3 channels -> (400, 400, 3) tensors.
    # this applies 32 convolution filters of size 3x3 each.
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(16, 16, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(neuron, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    adam = Adam(lr= learning_rate, beta_1=beta_1,beta_2=beta_2, epsilon=epsilon, decay=decay)
    model.compile(loss='categorical_crossentropy', optimizer=adam)

    return model

def fit_and_evaluate_model(model, x_train, y_train, x_test, y_test):
    print("Fitting......")
    model.fit(x_train, y_train, epochs=10, batch_size=64, verbose=0)
    print("Evaluating......")
    score = model.evaluate(x_test, y_test, verbose=0)
    return score

def cross_validate(n_folds, learning_rates, neurons):
    #Initialize variables for hyper_params. Can extend with more hyper_params.
    #final_scores = []
    best_score = 100
    best_neuron = -1
    #best_beta = 0

    #Grid search over hyper_params
    #for beta1 in beta_1:
    #    print("BETA:",beta1)
    for neuron in neurons:
        print("\nNEURON: ",neuron)
        score = 0
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True)
        splitted_indices = skf.split(np.zeros(x_train.shape[0]), y_train.T[0])
        for i, (train, test) in enumerate(splitted_indices):
            print("\nRunning Fold", i + 1, "/", n_folds, ", with ", neuron, " neurons.")
            model = create_model(learning_rates, neuron) # Start with a brand new model.
            score += (fit_and_evaluate_model(model, x_train[train], y_train[train], x_train[test], y_train[test]))
        avg_score = score/n_folds
        print("AVG: ", avg_score)
        #final_scores.append(avg_score)
        if (avg_score < best_score):
            best_neuron = neuron
            best_score = avg_score
            #best_beta = beta1
    return best_score, best_neuron #best_beta


#Perform Cross Validation
best_score, best_neuron = cross_validate(K_FOLDS, LEARNING_RATES, NEURONS)
#print("SCORES", scores)
print("\n")
print("BEST NEURON: ", best_neuron, "WITH ASSOCIATED SCORE OF: ", best_score)