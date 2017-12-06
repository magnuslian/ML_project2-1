import os
from Helpers import helpers
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np

from Models import cnnModel
from Given import proHelpers, given
from Helpers import the_model


DATAPATH_TESTING = "C:\\Users\\magnu\\Documents\\NTNU\\3 (Utveksling EPFL)\\Machine Learning\\Prosjekt2\\ML_project2-1\\test_set_images\\"
NUM_TEST_IMAGES = 50


model = the_model.create_model()
model.load_weights("C:\\Users\\magnu\\Documents\\NTNU\\3 (Utveksling EPFL)\\Machine Learning\\Prosjekt2\\ML_project2-1\\Saved_Weights\\weights_BM.h5")

x_test = helpers.load_test_data(DATAPATH_TESTING, NUM_TEST_IMAGES)
x_test = helpers.normalize(x_test, x_test.shape[1], std=False)

# Make prediction and submission file
test_image_patches = helpers.create_patches(x_test, 0.25, 16)

print("Predicting...")
pred = model.predict(test_image_patches)

pred1 = helpers.make_pred(pred)
pred2 = np.reshape(pred1, (50, 38, 38))
pred3 = helpers.filterh(pred2)
pred4 = np.reshape(pred3, (72200, 1))
submission = helpers.create_submission_format()
given.create_csv_submission(submission, pred4, "cleaned.csv")

"""
The pros


submission_filename = 'submission.csv'
image_filenames = []
root = "C:\\Users\\magnu\\Documents\\NTNU\\3 (Utveksling EPFL)\\Machine Learning\\Prosjekt2\\ML_project2-1\\test_set_images"
for i in range(1, 51):
    image_filename = root + '/test_' + str(i) + '/test_' + str(i) + '.png'
    image_filenames.append(image_filename)

proHelpers.generate_submission(model, submission_filename, *image_filenames)

"""