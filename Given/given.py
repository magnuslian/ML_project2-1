# Helper functions
import matplotlib.image as mpimg
import numpy as np
from PIL import Image
import csv

def load_image(infilename):
    data = mpimg.imread(infilename)
    return data

#Used in concatenate_images
def img_float_to_uint8(img):
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg

# Concatenate an image and its groundtruth
def concatenate_images(img, gt_img):
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
        cimg = np.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = np.zeros((w, h, 3), dtype=np.uint8)
        gt_img8 = img_float_to_uint8(gt_img)
        gt_img_3c[:,:,0] = gt_img8
        gt_img_3c[:,:,1] = gt_img8
        gt_img_3c[:,:,2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = np.concatenate((img8, gt_img_3c), axis=1)
    return cimg

def img_crop(im, w, h):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            if is_2d:
                im_patch = im[j:j+w, i:i+h]
            else:
                im_patch = im[j:j+w, i:i+h, :]
            list_patches.append(im_patch)
    return list_patches

#Try with padding and stride
def img_crop2(im, w, h, stride, padding):
    #Crop images so that they work with padding
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_3d = len(im.shape) == 3
    if is_3d: #Normal images
        print("Går inn her, normalbilde")
        im = np.lib.pad(im, ((padding, padding), (padding, padding), (0, 0)), 'reflect')
        for i in range(padding, imgheight + padding, stride):
            for j in range(padding, imgwidth + padding, stride):
                im_patch = im[j - padding:j + w + padding, i - padding:i + h + padding, :]
                print(im_patch.shape)
                list_patches.append(im_patch)
    else: #Ground truth
        print("Går inn her, GT")
        for i in range(0, imgheight, h):
            for j in range(0, imgwidth, w):
                im_patch = im[j:j + w, i:i + h]
                list_patches.append(im_patch)
    return list_patches


def value_to_class(v, foreground_threshold):
    # percentage of pixels > 1 required to assign a foreground label to a patch
    df = np.sum(v)
    if df > foreground_threshold:
        return [1,0]
    else:
        return [0,1]


# Convert array of labels to an image

def label_to_img(imgwidth, imgheight, w, h, labels):
    im = np.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            im[j:j + w, i:i + h] = labels[idx]
            idx = idx + 1
    return im


def make_img_overlay(img, predicted_img):
    w = img.shape[0]
    h = img.shape[1]
    color_mask = np.zeros((w, h, 3), dtype=np.uint8)
    color_mask[:, :, 0] = predicted_img * 255

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img

def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id': r1, 'Prediction': int(r2)})