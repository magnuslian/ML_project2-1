import numpy as np

from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import History, ModelCheckpoint
hist = History()



from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score, GridSearchCV

from Helpers import helpers


DATAPATH_TRAINING = "training/"
FILEPATH_SAVE = "model/"
NUM_TRAIN_IMAGES = 20
K_FOLDS = 4

#Initialize hyperparameters to be tested
LEARNING_RATES = np.logspace(-6,0,7)
BETA_1 = [0.7,0.9]
BATCH_SIZES = [16, 32, 64, 128]
NUMBER_OF_NEURONS = [16, 64, 128, 256]
DROPOUTS = [0, 0.2]

#Load training data
x_train = helpers.extract_data(DATAPATH_TRAINING, NUM_TRAIN_IMAGES)
y_train = helpers.extract_labels(DATAPATH_TRAINING, NUM_TRAIN_IMAGES)


#Normalizing the training data
x_train = np.reshape(x_train, (NUM_TRAIN_IMAGES*160000, 3))
x_train = helpers.normalize(x_train,std=False)
x_train = np.reshape(x_train, (NUM_TRAIN_IMAGES*625,16,16,3))


def create_model(neuron):
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

    adam = Adam(lr= 0.001, beta_1=0.7,beta_2=0.999,epsilon=1e-8,decay=0)
    print("Compiling model......")
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def fit_and_evaluate_model(model, x_train, y_train, x_val, y_val, checkpointer):
    print("Fitting......")
    model.fit(x_train, y_train, epochs=50, batch_size=64, verbose=1, validation_data=(x_val, y_val), callbacks=[checkpointer, hist])
    print(hist.history)
    return hist.history['val_loss']

def cross_validate(n_folds, neurons):
    #Initialize variables for hyper_params. Can extend with more hyper_params.
    best_loss = 100
    best_neuron = -1
    best_epoch = -1
    #Grid search over hyper_param
    counter = 0
    best_model = 0
    for neuron in neurons:
        val_losses = []
        counter += 1
        filepath = "weights/model" + str(counter) + ".h5"
        print("Neurons:", neuron)
        skf = StratifiedKFold(n_folds, shuffle=True)
        splitted_indices = skf.split(np.zeros(x_train.shape[0]), y_train.T[0])
        for i, (train, val) in enumerate(splitted_indices):
            print("\nRunning Fold", i + 1, "/", n_folds)
            model = create_model(neuron) # Start with a brand new model.
            checkpointer = ModelCheckpoint(filepath=filepath, verbose=1, save_best_only=True)
            val_losses.append(fit_and_evaluate_model(model, x_train[train], y_train[train], x_train[val], y_train[val],checkpointer))
        avg_val_losses = np.mean(val_losses, axis=0)
        val_loss = min(avg_val_losses)
        epoch = (np.where(avg_val_losses == val_loss))
        if (val_loss < best_loss):
            best_model = counter
            best_loss = val_loss
            best_neuron = neuron
            best_epoch = epoch
    return best_loss, best_neuron, best_model, best_epoch


#Perform Cross Validation
best_loss, best_neuron, best_model, best_epoch = cross_validate(K_FOLDS, NUMBER_OF_NEURONS)

print("BEST NEURON: ", best_neuron, " BEST LOSS:", best_loss)

print("BEST MODEL: ", best_model, "BEST EPOCH: ", best_epoch)