"""
Postprocessing of our predictions.
"""

import numpy as np


# Initialize some global variables for the road filters.
LENGTH = 8              # The length of each filter to be applied.
BACKGROUND_WIDTH = 3    # The width of each "background filter" to be applied.
RATIO = 0.25            # The threshold to determine if the background zone is considered background or not.


def filterh(images):
    """Applies multiple filters of different kinds to our prediction of the test set's ground truth images.

    :param images: Ground truth images to be filtered
    :return: The ground truth images after filtering
    """
    for nimage in range(images.shape[0]):
        image = images[nimage]

        # Three island filters
        image = islandfilter(image)
        image = islandfilter(image)
        image = islandfilter(image)

        # Clearing the roads before road filters
        image = straightroadhor(image)
        image = straightroadver(image)

        # Cleaning the sidewalks near the roads in the 4 directions
        image = roadhordown(image)
        image = roadhorup(image)
        image = roadverleft(image)
        image = roadverright(image)

        # Transform all the patches of a big road into "road".
        image = bigroad(image)

        images[nimage] = image

    return images


def islandfilter(image):
    """Applies filters of different size to transform road patches surrounded by background, and vice versa.

    :param image: Ground truth images of size  to be filtered
    :return: The ground truth images after filtering
    """
    # Avoid dimension problems
    padimage = np.lib.pad(image, ((2, 3), (2, 3)), 'reflect')

    # All the filters, of size 4x4, we use to select which patches we want to take into account.
    filter1 = np.array(((1, 1, 1, 0), (1, 0, 1, 0), (1, 1, 1, 0), (0, 0, 0, 0)))
    filter21 = np.array(((1, 0, 1, 0), (0, 0, 0, 0), (1, 0, 1, 0), (0, 0, 0, 0)))
    filter22 = np.array(((0, 1, 0, 0), (1, 0, 1, 0), (0, 1, 0, 0), (0, 0, 0, 0)))
    filter3 = np.array(((1, 1, 1, 1), (1, 0, 0, 1), (1, 1, 1, 1), (0, 0, 0, 0)))
    filter4 = np.array(((1, 1, 1, 0), (1, 0, 1, 0), (1, 0, 1, 0), (1, 1, 1, 0)))
    filter5 = np.array(((1, 1, 1, 1), (1, 0, 0, 1), (1, 0, 0, 1), (1, 1, 1, 1)))

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Take a subimage 4x4 where the patch (i,j) is in the position (2,2)
            subim = padimage[i + 1:i + 5, j + 1:j + 5]

            # If the patch is road and then it meets the condition, change into a background
            if image[i, j] == 1:
                # changes islands of type 1x1
                if np.sum(subim * filter1) == 0:
                    image[i, j] = 0
                # island 1x1
                if np.sum(subim * filter21) == 1 and np.sum(subim * filter22) == 0:
                    image[i, j] = 0
                # island 2x1
                if np.sum(subim * filter3) == 0:
                    image[i, j] = 0
                # island 1x2
                if np.sum(subim * filter4) == 0:
                    image[i, j] = 0
                # island 2x2
                if np.sum(subim * filter5) == 0:
                    # To avoid the dimensions problems when i or j is equal to image.shape[0].
                    try:
                        image[i:i + 2, j:j + 2] = np.array(((0, 0), (0, 0)))
                    except:
                        image[i, j] = 0
            # If the patch is background
            if image[i, j] == 0:
                # If all the surroundings are road, change into a road
                if np.sum(subim * filter1) == 8:
                    image[i, j] = 1
    return image


def straightroadhor(image):
    """Applies vector filters to check if a certain amount of the filter is road or background.
    If so, change the labels correspondingly. Applies horizntally.

    :param image: Ground truth images of size  to be filtered
    :return: The ground truth images after filtering
    """
    # Vectors of roads that we want to assign in the image.
    road1 = np.ones((9))
    road2 = np.ones((20))

    for j in range(image.shape[1]):
        # The optimum points to filter the image. We don't want to repeat the filter in the same place.
        for i in (0, 9, 18, 27):
            subim = image[i:i + 11, j]
            # If one or zero are background, change the values to road (except the extreme patches)
            if np.sum(subim) > 8:
                image[i + 1:i + 10, j] = road1

        # Points to apply the filter. We have been more strict in the condition so we can repeat patches.
        for i in (0, 6, 12, 16):
            subim2 = image[i:i + 22, j]
            if np.sum(subim2) > 19:
                image[i + 1:i + 21, j] = road2
    return image


def straightroadver(image):
    """The same function as above, but now we pass through the image with vertical filters instead of horizontal.

    :param image: Ground truth images of size  to be filtered
    :return: The ground truth images after filtering
    """

    road1 = np.ones((1, 9))
    road2 = np.ones((1, 20))

    for i in range(image.shape[0]):
        for j in (0, 9, 18, 27):
            subim = image[i, j:j + 11]
            if np.sum(subim) > 8:
                image[i, j + 1:j + 10] = road1

        for j in (0, 6, 12, 16):
            subim2 = image[i, j:j + 22]
            if np.sum(subim2) > 19:
                image[i, j + 1:j + 21] = road2
    return image


def roadhordown(image):
    """Applies two filters. One of size (LENGTHx1) to detect the road,
    and one of size (LENGTHxBACKGROUND_WIDTH) to detect background next to the road.

    :param image: Ground truth images of size  to be filtered
    :return: The ground truth images after filtering
    """
    # Pass through all the patches within boundary
    for i in range(image.shape[0] - LENGTH):
        # Variable to know if we were in a road or not in the last iteration.
        isroad = False
        for j in range(image.shape[1] - (BACKGROUND_WIDTH)):

            # Take a subimage of the bg
            bg = image[i:i + LENGTH, j:j + BACKGROUND_WIDTH]
            road = image[i:i + LENGTH, j + BACKGROUND_WIDTH:j + BACKGROUND_WIDTH + 1]
            mbg = np.mean(bg)
            mroad = np.mean(road)

            if not isroad:
                # If there is more than RATIO black patches in bg and all road is white,
                # change the patches (n,m) of bg to 0 except if all the patches in this column are road
                # (to avoid the elimination of the intersections).
                if mbg < RATIO and mroad == 1:
                    for n in range(bg.shape[0]):
                        for m in range(bg.shape[1]):
                            pa = bg[n, m]
                            if pa == 1 and np.sum(bg[n, :]) != BACKGROUND_WIDTH:
                                bg[n, m] = 0
                    isroad = True
            else:
                # Verify if we are still in a road.
                if mroad != 1.0:
                    isroad = False
            image[i:i + LENGTH, j:j + BACKGROUND_WIDTH] = bg

    return image


# The following three functions applies the same function as above, but in the other 3 directions.

def roadhorup(image):

    for i in range(image.shape[0] - LENGTH):
        isroad = False
        for j in range(image.shape[1] - (BACKGROUND_WIDTH) - 1, -1, -1):

            road = image[i:i + LENGTH, j:j + 1]
            bg = image[i:i + LENGTH, j + 1:j + 1 + BACKGROUND_WIDTH]
            mgb = np.mean(bg)
            mroad = np.mean(road)

            if not isroad:
                if mgb < RATIO and mroad == 1:
                    for n in range(bg.shape[0]):
                        for m in range(bg.shape[1]):
                            pa = bg[n, m]
                            if pa == 1 and np.sum(bg[n, :]) != BACKGROUND_WIDTH:
                                bg[n, m] = 0
                    isroad = True
            else:
                if mroad != 1.0:
                    isroad = False

            image[i:i + LENGTH, j + 1:j + 1 + BACKGROUND_WIDTH] = bg

    return image


def roadverright(image):

    for j in range(image.shape[1] - LENGTH):
        isroad = False

        for i in range(image.shape[0] - (BACKGROUND_WIDTH)):

            bg = image[i:i + BACKGROUND_WIDTH, j:j + LENGTH]
            road = image[i + BACKGROUND_WIDTH:i + BACKGROUND_WIDTH + 1, j:j + LENGTH]
            mg1 = np.mean(bg)
            mroad = np.mean(road)

            if not isroad:
                if mg1 < RATIO and mroad == 1:
                    for n in range(bg.shape[0]):
                        for m in range(bg.shape[1]):
                            pa = bg[n, m]
                            if pa == 1 and np.sum(bg[:, m]) != BACKGROUND_WIDTH:
                                bg[n, m] = 0
                    isroad = True
            else:
                if mroad != 1.0:
                    isroad = False

            image[i:i + BACKGROUND_WIDTH, j:j + LENGTH] = bg

    return image


def roadverleft(image):

    for j in range(image.shape[1] - LENGTH):
        isroad = False
        for i in range(image.shape[0] - (BACKGROUND_WIDTH) - 1, 0, -1):

            road = image[i:i + 1, j:j + LENGTH]
            bg = image[i + 1:i + 1 + BACKGROUND_WIDTH, j:j + LENGTH]
            mg1 = np.mean(bg)
            mroad = np.mean(road)
            if not isroad:
                if mg1 < RATIO and mroad == 1:
                    for n in range(bg.shape[0]):
                        for m in range(bg.shape[1]):
                            pa = bg[n, m]
                            if pa == 1 and np.sum(bg[:, m]) != BACKGROUND_WIDTH:
                                bg[n, m] = 0
                    isroad = True
            else:
                if mroad != 1.0:
                    isroad = False

            image[i + 1:i + 1 + BACKGROUND_WIDTH, j:j + LENGTH] = bg

    return image


def bigroad(image):
    """Applies a filter to correct bigger roads that crosses whole images.

    :param image: Ground truth images of size  to be filtered
    :return: The ground truth images after filtering
    """

    # Avoid dimensions problems
    padimage = np.lib.pad(image, ((1, 1), (1, 1)), 'constant', constant_values=((0, 0), (0, 0)))

    hor = np.sum(padimage, axis=0)
    ver = np.sum(padimage, axis=1)
    allh = np.ones(38)
    allv = np.ones((1, 38))

    # If there are two rows or columns next to each other,
    # and one has less than 3 background patches and the other less than 6,
    # change them into a road.
    for i in range(38):
        if ver[i] > 35 and ver[i - 1] > 32:
            padimage[i] = allv
            padimage[i - 1] = allv
        if ver[i] > 35 and ver[i + 1] > 32:
            padimage[i] = allv
            padimage[i + 1] = allv
        if hor[i] > 35 and hor[i - 1] > 32:
            padimage[:, i] = allh
            padimage[:, i - 1] = allh
        if hor[i] > 35 and hor[i + 1] > 32:
            padimage[:, i] = allh
            padimage[:, i + 1] = allh

    return padimage[1:-1, 1:-1]