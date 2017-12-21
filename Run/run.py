import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np

from Helpers import helpers, given, post_processing
from Model import model as m1

WINDOW_SIZE = 32
PADDING = (WINDOW_SIZE - 16) // 2

# Insert your datapath for test images here.
# Must be on the same form as in download file, and should end with "test_set_images\\"
DATAPATH_TESTING = "Datapath"
NUM_TEST_IMAGES = 50
FILEPATH_WEIGHTS = 'weights.h5'


# Load test data and create the correct format
imgs = helpers.load_test_data(DATAPATH_TESTING, NUM_TEST_IMAGES)
x_test = helpers.zero_mean(imgs, 608)
img_patches = helpers.create_patches_test_data(x_test, 16, 16, PADDING)

# Creating model and loading weights
model = m1.create_model()
model.load_weights(FILEPATH_WEIGHTS)

print("Predicting...")
softmax_pred = model.predict(img_patches) #(72200,2)
model_prediction = helpers.make_pred(softmax_pred) #(72200,1)

reshaped_prediction = np.reshape(model_prediction, (50,38,38))
postprocessed = post_processing.filterh(reshaped_prediction)
prediction = np.reshape(postprocessed, (72200,1))

submission = helpers.create_submission_format()
given.create_csv_submission(submission, prediction, "submission_team_fire.csv")