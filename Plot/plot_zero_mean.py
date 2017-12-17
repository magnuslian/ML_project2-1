"""A plot of the training data.

This is the script which created the plots of the data we did in the exploratory data analysis.
We plot three graphs, one for the Red channel, one for the Green and one for the Blue.
The y-axis is the mean of the channel per image (x-axis)

We did two plots: One before zero-mean, and one after
As an example here, we provide the plot before zero-mean.
To get the plot with zero_mean, remove the tag on line 22.
"""
import numpy as np
import matplotlib.pyplot as plt

from Helpers import helpers


DATAPATH_TRAINING = "datapath" # INSERT DATAPATH OF TRAINING DATA

NUM_OF_IMGS = 100

imgs, _ = helpers.load_training_data(DATAPATH_TRAINING, NUM_OF_IMGS)
#imgs = helpers.zero_mean(imgses,400)

red_array = []
green_array = []
blue_array = []

# Each of the arrays above should consist of 100 values after the for-loop.
for img in imgs:
    reshaped = np.reshape(img,(400*400,3))
    red_array.append(np.mean(reshaped[:,0])) #appends the mean of all the R values for all the pixels in that image.
    green_array.append(np.mean(reshaped[:,1]))
    blue_array.append(np.mean(reshaped[:,2]))

x_axis = np.arange(1, NUM_OF_IMGS +1)

#plot before zero_mean
plt.plot(x_axis, red_array, color='red')
plt.plot(x_axis, green_array, color= 'green')
plt.plot(x_axis, blue_array, color = 'blue')
plt.xlabel('Image')
plt.ylabel('Mean of channel')
plt.axis([0, NUM_OF_IMGS+1,-0.20,0.55])
plt.show()