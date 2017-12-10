import os
from Helpers import helpers
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np

from Given import proHelpers, given
from Helpers import the_model, postProcessing


PADDING = 4


DATAPATH_TESTING = "C:\\Users\\magnu\\Documents\\NTNU\\3 (Utveksling EPFL)\\Machine Learning\\Prosjekt2\\ML_project2-1\\test_set_images\\"
NUM_TEST_IMAGES = 50


model = the_model.create_model()
model.load_weights("C:\\Users\\magnu\\Documents\\NTNU\\3 (Utveksling EPFL)\\Machine Learning\\Prosjekt2\\ML_project2-1\\Saved_Weights\\weights_win24.h5")

imgs = helpers.load_test_data(DATAPATH_TESTING, NUM_TEST_IMAGES)
x_test = helpers.normalize(imgs, 608, std=False)
print("Creating patches...")
img_patches = proHelpers.create_patches(x_test, 16, 16, PADDING)

print("Predicting...")
pred = model.predict(img_patches)
pred1 = helpers.make_pred(pred)

"""
pred2 = np.reshape(pred1, (50, 38, 38))
pred3 = postProcessing.filterh(pred2)
pred4 = np.reshape(pred3, (72200, 1))
"""
submission = helpers.create_submission_format()
given.create_csv_submission(submission, pred1, "win24_0,3.csv")
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