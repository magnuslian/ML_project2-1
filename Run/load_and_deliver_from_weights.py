import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from Given import given
from Helpers import helpers
from Models import the_model

WINDOW_SIZE = 32
PADDING = (WINDOW_SIZE - 16) // 2


DATAPATH_TESTING = "C:\\Users\\magnu\\Documents\\NTNU\\3 (Utveksling EPFL)\\Machine Learning\\Prosjekt2\\ML_project2-1\\test_set_images\\"
NUM_TEST_IMAGES = 50


model = the_model.create_model()
model.load_weights("C:\\Users\\magnu\\Documents\\NTNU\\3 (Utveksling EPFL)\\Machine Learning\\Prosjekt2\\ML_project2-1\\Saved_Weights\\weights_best.h5")

imgs = helpers.load_test_data(DATAPATH_TESTING, NUM_TEST_IMAGES)
x_test = helpers.zero_mean(imgs, 608)
print("Creating patches...")
img_patches = helpers.create_patches_test_data(x_test, 16, 16, PADDING)

print("Predicting...")
pred = model.predict(img_patches)
pred1 = helpers.make_pred(pred)

"""
pred2 = np.reshape(pred1, (50, 38, 38))
pred3 = postProcessing.filterh(pred2)
pred4 = np.reshape(pred3, (72200, 1))
"""

submission = helpers.create_submission_format()
given.create_csv_submission(submission, pred1, "win32_greyscale.csv")