import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from keras.models import load_model

from Helpers import helpers
from Models import cnnModel
from Given import proHelpers



DATAPATH_TESTING = "C:\\Users\\magnu\\Documents\\NTNU\\3 (Utveksling EPFL)\\Machine Learning\\Prosjekt2\\ML_project2-1\\test_set_images\\"
NUM_TEST_IMAGES = 50


model = cnnModel.CnnModel()
model.load("C:\\Users\\magnu\\Documents\\NTNU\\3 (Utveksling EPFL)\\Machine Learning\\Prosjekt2\\ML_project2-1\\Saved_Weights\\weights.h5")

#x_test = helpers.load_test_data(DATAPATH_TESTING, NUM_TEST_IMAGES)
#x_test = helpers.normalize(x_test, x_test.shape[1], std=False)

#model.prediction(x_test)

"""
The pros
"""

submission_filename = 'submission.csv'
image_filenames = []
root = "C:\\Users\\magnu\\Documents\\NTNU\\3 (Utveksling EPFL)\\Machine Learning\\Prosjekt2\\ML_project2-1\\test_set_images"
for i in range(1, 51):
    image_filename = root + '/test_' + str(i) + '/test_' + str(i) + '.png'
    image_filenames.append(image_filename)

proHelpers.generate_submission(model, submission_filename, *image_filenames)