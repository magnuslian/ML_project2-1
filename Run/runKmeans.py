from Helpers import helpers
from Given import given
from sklearn import cluster
import numpy as np


DATAPATH_TRAINING = "C:\\Users\\magnu\\Documents\\NTNU\\3 (Utveksling EPFL)\\Machine Learning\\Prosjekt2\\Data\\training\\"
DATAPATH_TESTING = "C:\\Users\\magnu\\Documents\\NTNU\\3 (Utveksling EPFL)\\Machine Learning\\Prosjekt2\\Data\\test_set_images\\"

imgs_train, gt_imgs_train = helpers.load_training_data(DATAPATH_TRAINING)
X_train, Y_train = helpers.create_patches_and_arrays(imgs_train, 0.25, gt_imgs_train)

imgs_test = helpers.load_test_data(DATAPATH_TESTING)
X_test = helpers.create_patches_and_arrays(imgs_test, 0.25)


# K-means as Machine Learning algorithm
kmeans = cluster.KMeans(n_clusters=2, init="k-means++", n_init=10, max_iter=300, tol=0.001)
kmeans.fit(X_train, Y_train)
pred = kmeans.predict(X_test)



submission = helpers.create_submission_from_prediction(pred)

print("True positive rate: " + helpers.true_positive_rate(pred,Y_train))

given.create_csv_submission(submission, pred, "low TRP")