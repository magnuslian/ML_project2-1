import numpy as np
import matplotlib.pyplot as plt

from Helpers.helpers import load_training_data, normalize


datapath = "C:\\Users\\magnu\\Documents\\NTNU\\3 (Utveksling EPFL)\\Machine Learning\\Prosjekt2\\Data\\training\\"

NUM_OF_IMGS = 100

imgs, _ = load_training_data(datapath, NUM_OF_IMGS)

# 3D plot
"""
fig = pyplot.figure()
axis = fig.add_subplot(1, 1, 1, projection="3d") # 3D plot with scalar values in each axis

r, g, b = list(imgs.getdata(0)), list(imgs.getdata(1)), list(imgs.getdata(2))

axis.scatter(r, g, b, c="#ff0000", marker="o")
axis.set_xlabel("Red")
axis.set_ylabel("Green")
axis.set_zlabel("Blue")
pyplot.show()"""

red_array = []
green_array = []
blue_array = []

#imgs = normalize(imgses,400,std=False)

# Each of the arrays above should consist of 100 values after the for-loop.
for img in imgs:
    reshaped = np.reshape(img,(400*400,3))
    red_array.append(np.mean(reshaped[:,0])) #appends the mean of all the R values for all the pixels.
    green_array.append(np.mean(reshaped[:,1]))
    blue_array.append(np.mean(reshaped[:,2]))

x_axis = np.arange(1, NUM_OF_IMGS +1)

#plot before normalizing
plt.plot(x_axis, red_array, color='red')
plt.plot(x_axis, green_array, color= 'green')
plt.plot(x_axis, blue_array, color = 'blue')
plt.xlabel('Image')
plt.ylabel('Mean of channel')
#plt.title('The mean of each channel for each picture before normalization')
plt.axis([0, NUM_OF_IMGS+1,-0.20,0.55])
plt.show()