import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from Helpers import helpers



DATAPATH_TRAINING = "C:\\Users\\magnu\\Documents\\NTNU\\3 (Utveksling EPFL)\\Machine Learning\\Prosjekt2\\ML_project2-1\\training\\"
DATAPATH_TESTING = "C:\\Users\\magnu\\Documents\\NTNU\\3 (Utveksling EPFL)\\Machine Learning\\Prosjekt2\\ML_project2-1\\test_set_images\\"

NUM_TRAIN_IMAGES = 100
NUM_TEST_IMAGES = 50

x_train, y_train = helpers.load_training_data(DATAPATH_TRAINING, NUM_TRAIN_IMAGES)

model.train(x_train, y_train, save_weights=True)

x_test = helpers.load_test_data(DATAPATH_TESTING, NUM_TEST_IMAGES)
x_test = helpers.normalize(x_test, x_test.shape[1], std=False)

model.prediction(x_test)