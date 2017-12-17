import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from Helpers import helpers, given
from Model import the_model

WINDOW_SIZE = 32
PADDING = (WINDOW_SIZE - 16) // 2


DATAPATH_TESTING = "C:\\Users\\magnu\\Documents\\NTNU\\3 (Utveksling EPFL)\\Machine Learning\\Prosjekt2\\Data\\test_set_images\\"
NUM_TEST_IMAGES = 50


model = the_model.create_model()
model.load_weights('weights_best.h5')
model.summary()

imgs = helpers.load_test_data(DATAPATH_TESTING, NUM_TEST_IMAGES)
x_test = helpers.zero_mean(imgs, 608)
print("Creating patches...")
img_patches = helpers.create_patches_test_data(x_test, 16, 16, PADDING)

print("Predicting...")
pred = model.predict(img_patches)
pred1 = helpers.make_pred(pred)

#INSERT POST PROCESSING HERE

submission = helpers.create_submission_format()
given.create_csv_submission(submission, pred1, "win32_greyscale.csv")