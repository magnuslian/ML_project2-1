#!/usr/bin/python
from PIL import Image
import math
import numpy as np

from Helpers import helpers
from Given import given
import matplotlib.pyplot as plt

label_file = 'C:\\Users\\magnu\\Documents\\NTNU\\3 (Utveksling EPFL)\\Machine Learning\\Prosjekt2\\ML_project2-1\\Helpers\\test.csv'
test_imgs =  "C:\\Users\\magnu\\Documents\\NTNU\\3 (Utveksling EPFL)\\Machine Learning\\Prosjekt2\\Data\\test_set_images\\"

imgs = helpers.load_test_data(test_imgs,50)

h = 16
w = h
imgwidth = int(math.ceil((600.0 / w)) * w)
imgheight = int(math.ceil((600.0 / h)) * h)
nc = 3


# Convert an array of binary labels to a uint8
def binary_to_uint8(img):
    rimg = (img * 255).round().astype(np.uint8)
    return rimg


def reconstruct_from_labels(image_id):
    im = np.zeros((imgwidth, imgheight), dtype=np.uint8)
    f = open(label_file)
    lines = f.readlines()
    image_id_str = '%.3d_' % image_id
    for i in range(1, len(lines)):
        line = lines[i]
        if not image_id_str in line:
            continue

        tokens = line.split(',')
        id = tokens[0]
        prediction = int(tokens[1])
        tokens = id.split('_')
        i = int(tokens[1])
        j = int(tokens[2])

        je = min(j + w, imgwidth)
        ie = min(i + h, imgheight)
        if prediction == 0:
            adata = np.zeros((w, h))
        else:
            adata = np.ones((w, h))

        im[j:je, i:ie] = binary_to_uint8(adata)

    cimg = given.concatenate_images(imgs[image_id-1], im)
    fig1 = plt.figure(figsize=(10, 10))  # create a figure with the default size

    Image.fromarray(cimg).save('prediction_' + '%.3d' % image_id + '.png')

    return im


for i in range(1, 51):
    reconstruct_from_labels(i)