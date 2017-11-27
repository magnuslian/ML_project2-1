import numpy as np
import matplotlib.pyplot as plt
import os
from Given import given


def load_training_data(datapath, number_of_images):
    # Load all images
    image_dir = datapath + "images/"
    files = os.listdir(image_dir)
    print("Loading " + str(number_of_images) + " images")
    imgs = [given.load_image(image_dir + files[i]) for i in range(number_of_images)]

    # Load all groundtruth images
    gt_dir = datapath + "groundtruth/"
    print("Loading " + str(len(files)) + " groundtruth images")
    gt_imgs = [given.load_image(gt_dir + files[i]) for i in range(number_of_images)]

    return imgs, gt_imgs


def sort_key(s):
    s, n = s.split('_')
    return s, int(n)

def load_test_data(datapath, number_of_images):
    image_dir_test = datapath
    files = os.listdir(image_dir_test)
    print("Loading " + str(number_of_images) + " images")
    files.sort(key=sort_key)

    folder = []
    for i in range(len(files)):
        folder.append(image_dir_test + files[i] + "\\" + files[i] + ".png")

    imgs_test = [given.load_image(folder[i]) for i in range(len(folder))]

    return imgs_test


def show_nth_image(imgs, gt_imgs, nth_image):
    cimg = given.concatenate_images(imgs[nth_image], gt_imgs[nth_image])
    fig1 = plt.figure(figsize=(10, 10))
    plt.imshow(cimg, cmap='Greys_r')


def create_patches_and_arrays(imgs, FOREGROUND_THRESHOLD, gt_imgs=[]):
    PATCH_SIZE = 16  # each patch is 16*16 pixels

    # Extract patches from input images
    img_patches = [given.img_crop(imgs[i], PATCH_SIZE, PATCH_SIZE) for i in range(len(imgs))]
    gt_patches = [given.img_crop(gt_imgs[i], PATCH_SIZE, PATCH_SIZE) for i in range(len(gt_imgs))]

    # Linearize list of patches
    img_patches = np.asarray([img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))])
    gt_patches = np.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])

    # Compute features (mean and variance) for each image patch
    x = np.asarray(
        [given.extract_features_2d(img_patches[i]) for i in range(len(img_patches))])
    # a len(img_patches)x2 matrix with the first column being the mean of the image,
    # and the second the variance of the image.

    if len(gt_imgs) != 0:
        # Compute array based on foreground threshold
        y = np.asarray(
            [given.value_to_class(np.mean(gt_patches[i]), FOREGROUND_THRESHOLD) for i in range(len(gt_patches))])
        # a list of 0's and 1's, depending on if the mean of the patch is greater than a given foreground_threshold,
        # which qualifies it to be a foreground patch (the index 1).
        return x, y

    return x


def create_submission_from_prediction(pred):
    submission = []
    con = 0
    for im in range(1, 51):
        for row in range(0, 608, 16):
            for col in range(0, 608, 16):
                c = str(im).zfill(3) + '_' + str(row) + "_" + str(col)
                con += 1
                submission.append(c)
    return submission

def true_positive_rate(pred,y):
    # Get non-zeros in prediction and grountruth arrays
    zn = np.nonzero(pred)[0]
    yn = np.nonzero(y)[0]

    # percentage of indices with 1's in both yn and zn
    TPR = (len(list(set(yn) & set(zn))) / float(len(pred))) * 100
    TPR_round = round(TPR, 2)
    return (str(TPR_round) + " %")