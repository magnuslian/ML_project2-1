import numpy as np
import tensorflow as tf
import keras as k

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.preprocessing import image

#from Prosjekt2 import helpers
#from Given import given

model = Sequential()

gen_flow = image.ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=False)

train_generator = gen_flow.flow_from_directory(
    "C:\\Users\\magnu\\Documents\\NTNU\\3 (Utveksling EPFL)\\Machine Learning\\Prosjekt2\\Data\\training/",
    target_size=(256,256),
    shuffle=False,
    classes=['images','groundtruth'],
    batch_size=32
    )


print(len(train_generator(0)))

model.fit_generator(

)

