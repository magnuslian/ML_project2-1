import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Cropping2D
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam
from keras.preprocessing import image

from Helpers import helpers
from Given import given

DATAPATH_TRAINING = "C:\\Users\\magnu\\Documents\\NTNU\\3 (Utveksling EPFL)\\Machine Learning\\Prosjekt2\\Data\\training\\"
DATAPATH_TESTING = "C:\\Users\\magnu\\Documents\\NTNU\\3 (Utveksling EPFL)\\Machine Learning\\Prosjekt2\\Data\\test_set_images\\"


gen_flow = image.ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=False)

train_generator = gen_flow.flow_from_directory(
    DATAPATH_TRAINING,
    target_size=(256,256),
    classes=['images','groundtruth'],
    batch_size=32,
    class_mode='binary'
    )

test_generator = gen_flow.flow_from_directory(
    DATAPATH_TESTING,
    target_size=(256,256),
    batch_size=32,
    class_mode='binary'
    )

model = Sequential()

model.add(Cropping2D(cropping=(16,16), input_shape=(256,256,3)))

# input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
# this applies 32 convolution filters of size 3x3 each.
model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
# shape
model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
# shape (4,4,64)

model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='softmax'))

adam = Adam(lr=0.001, beta_1=0.7,epsilon=1e-8,decay=0)
model.compile(loss='binary_crossentropy', optimizer=adam)

model.fit_generator(
    train_generator,
    epochs=2,
    steps_per_epoch=16,
    validation_data=test_generator,
    validation_steps=8
)
pred = model.predict_generator

print(pred)


pred1 = helpers.make_pred(pred)
submission = helpers.create_submission_format()
given.create_csv_submission(submission, pred1, "blabla.csv")