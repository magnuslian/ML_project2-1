import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from Given import given
from sklearn import preprocessing
from PIL import Image


IMG_PATCH_SIZE = 16
FOREGROUND_THRESHOLD = 0.25

def normalize(data, patch_size, mean = True, std = True):
    data1 = np.reshape(data, (data.shape[0] * patch_size * patch_size, 3))
    scaler = preprocessing.StandardScaler(copy=True, with_mean=mean, with_std=std)
    scaler.fit(data1)
    new_data=scaler.fit_transform(data1, y=None)
    out_data = np.reshape(new_data, (data.shape[0], patch_size, patch_size, 3))
    return out_data

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
    for i in range(len(files)):
        folder.append(image_dir_test + files[i] + "\\" + files[i] + ".png")

    imgs_test = [given.load_image(folder[i]) for i in range(len(folder))]

    return np.asarray(imgs_test)

def sort_key(s):
    s, n = s.split('_')
    return s, int(n)


def create_patches(imgs, foreground_threshold, patch_size, gt_imgs=[]):
    print("Creating patches of the input data...")
    # Extract patches from input images
    img_patches = [given.img_crop(imgs[i], patch_size, patch_size) for i in range(len(imgs))]
    gt_patches = [given.img_crop(gt_imgs[i], patch_size, patch_size) for i in range(len(gt_imgs))]

    # Linearize list of patches
    img_patches = np.asarray([img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))])
    gt_patches = np.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])

    if len(gt_imgs) != 0:
        # Compute array based on foreground threshold
        labels = np.asarray([given.value_to_class(np.mean(gt_patches[i]), foreground_threshold) for i in range(len(gt_patches))])
        # a list of 0's and 1's, depending on if the mean of the patch is greater than a given foreground_threshold,
        # which qualifies it to be a foreground patch (the index 1).
        return img_patches, labels.astype(np.float32)

    return img_patches


def calculate_score(pred, sol):
    if len(pred) != len(sol):
        raise ValueError("The length of the prediction does not match the length of the solution")
    sum = 0
    for i in range(len(pred)):
        if pred[i] == sol[i]:
            sum += 1
    score = round(sum/len(pred) * 100,2)
    print("Accuracy: " + score + "%")
    return score


def create_submission_format():
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

def make_pred(pred):
    out = []
    for p in pred:
        if p[0] <= 0.25:
            out.append(0)
        else:
            out.append(1)
    return out