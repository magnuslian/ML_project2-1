import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from keras.models import load_model

from Helpers import helpers
from Models import cnnModel



DATAPATH_TESTING = "C:\\Users\\magnu\\Documents\\NTNU\\3 (Utveksling EPFL)\\Machine Learning\\Prosjekt2\\ML_project2-1\\test_set_images\\"
NUM_TEST_IMAGES = 50


model = cnnModel.CnnModel()
model.load("C:\\Users\\magnu\\Documents\\NTNU\\3 (Utveksling EPFL)\\Machine Learning\\Prosjekt2\\ML_project2-1\\Saved_Weights\\weights.h5")

x_test = helpers.load_test_data(DATAPATH_TESTING, NUM_TEST_IMAGES)
print(x_test.shape)
x_test = helpers.normalize(x_test, x_test.shape[1], std=False)

model.prediction(x_test)