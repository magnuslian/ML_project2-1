import numpy as np
import tensorflow as tf
import keras as k

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.preprocessing import image

from Helpers import helpers
from Given import given


gen_flow = image.ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=False)

train_generator_img = gen_flow.flow_from_directory(
    "C:/Users/magnu/Documents/NTNU/3 (Utveksling EPFL)/Machine Learning/Prosjekt2/Data/training/images",
    target_size=(625,625),
    shuffle=False,
    )

train_generator_gt = gen_flow.flow_from_directory(
    "C:\\Users\\magnu\\Documents\\NTNU\\3 (Utveksling EPFL)\\Machine Learning\\Prosjekt2\\Data\\training\\groundtruth",
    target_size=(625,625),
    shuffle=False,
    )

print(train_generator_img)

print(train_generator_gt)