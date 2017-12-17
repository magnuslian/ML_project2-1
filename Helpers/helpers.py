import numpy as np
import os

from skimage.exposure import rescale_intensity

from Helpers import given


def load_training_data(datapath, number_of_images):
    """Load a number of training images and corresponding ground truths"""
    image_dir = datapath + "images/"
    files = os.listdir(image_dir)
    print("Loading " + str(number_of_images) + " images")
    imgs = np.asarray([given.load_image(image_dir + files[i]) for i in range(number_of_images)])

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

    imgs_test = np.asarray([given.load_image(folder[i]) for i in range(len(folder))])

    return imgs_test

def sort_key(input):
    """Sorts the image numerically instead of lexographically

    Used in load_test_data
    :param input: The filename of one test image
    :return: The first part of the filename as a token, and the image number
    """
    token1, image_number = input.split('_')

    return token1, int(image_number)


def zero_mean(data, window_size, std=False):
    """Zero-means the data, with possibility of standardizing as well"""
    data1 = np.reshape(data, (data.shape[0] * window_size * window_size, 3))
    data1 -= np.mean(data1, axis=0)
    if std:
        data1 /= np.std(data1, axis=0)
    out_data = np.reshape(data1, (data.shape[0], window_size, window_size, 3))
    return out_data


def create_patches_test_data(imgs, patch_size, stride, padding):
    """Creates patches of the test data so that it fits the weights from the model.

    E.g if we trained the model with input size = 32 x 32 x 3, these patches also have to be of that size.
    :param imgs: Test images
    :param patch_size: Size of patches on Kaggle. Should be 16
    :param stride: Distance between the start of one patch, to the next (to the right and below)
    :param padding: Padding of the image
    :return: Test images divided into patches that matches the window size from the model
    """
    # Extract patches from input images
    img_patches = [img_crop(imgs[i], patch_size, patch_size, stride, padding) for i in range(len(imgs))]

    # Linearize list of patches, code from tf_aerial_images.py
    img_patches = np.asarray([img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))])

    return img_patches


#
# M
def img_crop(image, width, height, stride, padding):
    """Crops a single input image into the wished size.

    Makes a crop of 16 x 16 from the image, and then pads it so that it matches the weights from the model.
    :param image: One test image
    :param width: Width of the patch. Should be 16.
    :param height: Height of the patch. Should be 16.
    :param stride: Distance between the start of one patch, to the next (to the right and below)
    :param padding: Padding of the image
    :return: A list of all the patches in the image correctly padded to match the window size of the model.
    """
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


def create_random_windows_of_training_data(x_train, y_train, num_windows_per_img, window_size):
    """Takes in whole images as arguments and creates num_windows_per_img per image

    :param x_train: The training data (Google Maps images).
    :param y_train: The training data (ground truth images).
    :param num_windows_per_img: Desired number of windows which should be created per image.
    :param window_size: Window size of each window created.
    :return: All the created windows and corresponding label.
    """
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


#
# .
def image_augmentation(window_x):
    """Manipulates the image to change the order of the pixels.

    The corresponding ground truth image will not change.
    :param window_x: The window to be manipulated.
    :return: The window after manipulation
    """
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


def make_pred(pred):
    """Makes the prediction of the correct format after model.predict

    E.g takes in [0.22, 0.78] and evaluates it to 0 (background).
    :param pred: A matrix of prediction on the form (num_patches, 2)
    :return: A list of predictions to the input matrix
    """
    out = []
    for p in pred:
        if p[0] <= 0.50:
            out.append(0) #Gives black
        else:
            out.append(1) #White
    return out