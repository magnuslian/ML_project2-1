# Road segmentation using Convolutional Neural Networks
#### Project 2, Machine Learning, EPFL, Fall 2017.

*Magnus Lysholm Lian, Jørgen Frost Bø, Raul Villalba Felix*

This project presents our solution to the road segmentation task. We implemented a convolutional neural network which
got us a Mean F score of 0,892 on Kaggle.

### External libraries

We have used two external libraries for this project:

- [Keras version 2.0.8](https://keras.io/#installation)
- [TensorFlow version 1.2.1](https://www.tensorflow.org/install/)

Keras uses TensorFlow as backend. Please be certain that both of these libraries are installed before running any code.

### Run

To save time to create the prediction file, we have already trained the model and saved
the weights in the file `weights.h5`. To get the prediction file, please run the `run.py`.
Keep in mind that you will have to provide a valid datapath for the test data to be able to run the file. The images
have to be put in the file `test_set_images` as they were when downloaded.

### Train

We have provided the file `train_model.py` for those who wish to run the model from scratch.
This will however take some time, depending on the computer setup.
Please provide the correct path for both training and test data to be able to run the file.

### Contents

- *Helpers:* Contains 5 files. `given.py` and `submission_to_mask.py` were given at the start of the project.
The first consists of the help methods we were given from start.
The second was used to create some of the masks in the report. `helpers.py` contains some necessary help methods that
we have created ourselves.
`post_processing.py` contains all the code we used for the
postprocessing part of our submission. `model&hyper_parameter_tuning.py` contains an example of how we
evaluated our model when searching for the best combination of hyper parameters. Note the code for plotting the
validation loss/accuracy per epoch which were used in the report. This file is runable.
- *Model:* Contains 2 files. `model.py` contains our model and is called in some other files.
'train_model.py' is a complete setup of how we trained our model. This file is runable.
- *Plots:* Contains 1 file. `plot_zero_mean.py` is the code for creating the plot of the initial data before and
after zero-mean.
- *Run:* Contains 2 files. `run.py` is the file which creates the submission for our best effort model.
`weights.h5` is the weights for this model.