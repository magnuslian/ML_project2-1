from PIL import Image
import math
import numpy as np

from Helpers import helpers, given

label_file = 'window32.csv'
    #'C:\\Users\\magnu\\Documents\\NTNU\\3 (Utveksling EPFL)\\Machine Learning\\Prosjekt2\\ML_project2-1\\Run\\win24_greyscale.csv'
test_imgs =  "C:\\Users\\magnu\\Documents\\NTNU\\3 (Utveksling EPFL)\\Machine Learning\\Prosjekt2\\Data\\test_set_images\\"

imgs = helpers.load_test_data(test_imgs,5)

h = 16
w = 16
imgwidth = int(math.ceil((600.0 / w)) * w)
imgheight = int(math.ceil((600.0 / h)) * h)
nc = 3


def binary_to_uint8(img):
    """Convert an array of binary labels to a uint8"""
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
        elif prediction == 1:
            adata = np.ones((w, h))
        else:
            adata = np.full((w, h), 0.5)
        im[j:je, i:ie] = binary_to_uint8(adata)

    cimg = given.concatenate_images(imgs[image_id - 1], im)
    Image.fromarray(cimg).save('0,3_' + '%.3d' % image_id + '.png')

    return im


for i in range(1,imgs.shape[0]+1):
    reconstruct_from_labels(i)