import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from Helpers import helpers
from Models import cnnModel



DATAPATH_TRAINING = 'training/'
DATAPATH_TESTING = 'test_set_images/'

NUM_TRAIN_IMAGES = 100
NUM_TEST_IMAGES = 50

x_train, y_train = helpers.load_training_data(DATAPATH_TRAINING, NUM_TRAIN_IMAGES)

model = cnnModel.initialize()
trained_model = cnnModel.train(model, x_train, y_train,save_model=True)



x_test = helpers.load_test_data(DATAPATH_TESTING, NUM_TEST_IMAGES)
x_test = helpers.normalize(x_test, x_test.shape[1], std=False)

cnnModel.prediction(trained_model,x_test)