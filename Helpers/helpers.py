import numpy as np
import os

from skimage.exposure import rescale_intensity

from Given import given


def load_training_data(datapath, number_of_images):
    # Load all images
    image_dir = datapath + "images/"
    files = os.listdir(image_dir)
    print("Loading " + str(number_of_images) + " images")
    imgs = np.asarray([given.load_image(image_dir + files[i]) for i in range(number_of_images)])

    # Load all groundtruth images
    gt_dir = datapath + "groundtruth/"
    print("Loading " + str(number_of_images) + " groundtruth images")
    gt_imgs = np.asarray([given.load_image(gt_dir + files[i]) for i in range(number_of_images)])

    return imgs, gt_imgs

def load_test_data(datapath, number_of_images):
    image_dir_test = datapath
    files = os.listdir(image_dir_test)
    print("Loading " + str(number_of_images) + " images")
    files.sort(key=sort_key)

    folder = []
    for i in range(number_of_images):
        folder.append(image_dir_test + files[i] + "\\" + files[i] + ".png")

    imgs_test = [given.load_image(folder[i]) for i in range(len(folder))]

    return np.asarray(imgs_test)

# Sorts the image correctly on number instead of lexographic
# Used in load_test_data
def sort_key(s):
    s, n = s.split('_')
    return s, int(n)


# Zero-means the data, with possibility of standardizing as well
def zero_mean(data, window_size, std=False):
    data1 = np.reshape(data, (data.shape[0] * window_size * window_size, 3))
    data1 -= np.mean(data1, axis=0)
    if std:
        data1 /= np.std(data1, axis=0)
    out_data = np.reshape(data1, (data.shape[0], window_size, window_size, 3))
    return out_data


# Creates patches of the test data so that it fits the weights from the model
# E.g if we trained the model with input size = 32 x 32 x 3,
# these patches also have to be of that size.
def create_patches_test_data(imgs, patch_size, stride, padding):
    # Extract patches from input images
    img_patches = [img_crop(imgs[i], patch_size, patch_size, stride, padding) for i in range(len(imgs))]

    # Linearize list of patches, code from tf_aerial_images.py
    img_patches = np.asarray([img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))])

    return img_patches


# Crops a single input image into the wished size.
# Makes a crop of (16, 16) from the image, and then pads it so that it matches
# the weights from the model.
def img_crop(image, width, height, stride, padding):
    list_patches = []
    imgwidth = image.shape[0]
    imgheight = image.shape[1]
    if len(image.shape) == 3:
        im = np.lib.pad(image, ((padding, padding), (padding, padding), (0, 0)), 'reflect')
        for h in range(padding, imgheight + padding, stride):
            for w in range(padding, imgwidth + padding, stride):
                im_patch = im[w - padding:w + width + padding, h - padding:h + height + padding, :]
                list_patches.append(im_patch)
    return list_patches


# Takes in whole images as arguments and creates num_windows_per_img per image
def create_random_patches_of_training_data(x_train, y_train, num_windows_per_img, window_size):
    x_ptrain = np.empty((x_train.shape[0] * num_windows_per_img, window_size, window_size, 3))
    y_ptrain = np.empty((y_train.shape[0] * num_windows_per_img, 2))

    # Iterate over every image in the training set
    for pic in range(x_train.shape[0]):
        # Extract how many windows we want from each image
        for iter in range(0, num_windows_per_img):
            width = x_train[pic].shape[0]
            height = x_train[pic].shape[1]

            # Random window from the image
            randomw = np.random.randint(0, width - window_size + 1)
            randomh = np.random.randint(0, height - window_size + 1)
            subimage_x = x_train[pic][randomw:randomw + window_size, randomh:randomh + window_size]
            subimage_y = y_train[pic][randomw:randomw + window_size, randomh:randomh + window_size]

            #Image augmentation on x, and create the value of corresponding y from ground truth.
            subimage_x = image_augmentation(subimage_x)
            subimage_y = given.value_to_class(np.mean(subimage_y))

            x_ptrain[pic*num_windows_per_img + iter] = subimage_x
            y_ptrain[pic*num_windows_per_img + iter] = subimage_y
        print("Finished processing ", pic + 1)

    return x_ptrain, y_ptrain


# Building a bigger data set by manipulating the training data
# Takes windows as input data (e.g 32 x 32)
# Creates as many versions of the given windows as wanted
def increase_training_set(x_train, y_train, num_windows_per_window, window_size):
    x_ptrain = np.empty((x_train.shape[0] * num_windows_per_window, window_size, window_size, 3))
    y_ptrain = np.empty((y_train.shape[0] * num_windows_per_window, 2))

    for iter in range(0, num_windows_per_window):
        # Create e.g 8 versions of each patch
        for patch in range(x_train.shape[0]):
            subimage_x = x_train[patch]
            subimage_y = y_train[patch]

            subimage_x = image_augmentation(subimage_x)

            x_ptrain[patch * num_windows_per_window + iter] = subimage_x
            y_ptrain[patch * num_windows_per_window + iter] = subimage_y
        print("Done creating version ", iter + 1, " of the patches")
    return x_ptrain, y_ptrain


# Manipulates the image to change the order of the pixels.
# The corresponding ground truth image is the same.
def image_augmentation(window_x):
    # Contrast stretching
    if np.random.randint(2) == 0:
        window_x = rescale_intensity(window_x)

    # Random flip vertically
    if np.random.randint(2) == 0:
        window_x = np.flipud(window_x)

    # Random flip horizontally
    if np.random.randint(2) == 0:
        window_x = np.fliplr(window_x)

    # Random rotation in steps of 90°
    num_rot = np.random.randint(4)
    window_x = np.rot90(window_x, num_rot)

    return window_x


def create_submission_format():
    submission = []
    for im in range(1, 51):
        for row in range(0, 608, 16):
            for col in range(0, 608, 16):
                c = str(im).zfill(3) + '_' + str(row) + "_" + str(col)
                submission.append(c)
    return submission


# Makes the prediction of the correct format after model.predict
# Takes in a matrix of prediction on the form (num_patches, 2)
# E.g takes in [0.22, 0.78] and evaluates it to 0 (background).
def make_pred(pred):
    out = []
    for p in pred:
        if p[0] <= 0.50:
            out.append(0) #Gives black
        else:
            out.append(1) #White
    return out