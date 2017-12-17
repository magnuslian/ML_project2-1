import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from Given import given
from Helpers import helpers
from Models import the_model

WINDOW_SIZE = 32
PADDING = (WINDOW_SIZE - 16) // 2


DATAPATH_TESTING = "Datapath" #Insert your datapath for test images here. Must be on the form from download file.
NUM_TEST_IMAGES = 50
FILEPATH_WEIGHTS = 'weights_best.h5'


# Load the test data
imgs = helpers.load_test_data(DATAPATH_TESTING, NUM_TEST_IMAGES)
x_test = helpers.zero_mean(imgs, 608)
img_patches = helpers.create_patches_test_data(x_test, 16, 16, PADDING)

# Creating model and loading weights
model = the_model.create_model()
model.load_weights(FILEPATH_WEIGHTS)

print("Predicting...")
pred = model.predict(img_patches)
pred1 = helpers.make_pred(pred)

submission = helpers.create_submission_format()
given.create_csv_submission(submission, pred1, "submission_team_fire.csv")