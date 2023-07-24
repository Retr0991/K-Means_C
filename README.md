# K-Means_C
Implementation of the K-Means Clustering algorithm using C

This algorithm uses the K-Means++ alg to set the initial centroids

Persisting Problems:

•	The code is giving entirely accurate centroids now as I compared it using the sklearn library in Python. But the data points themselves are a bit problematic as they are being allotted to the wrong clusters  8 – 10% of the time as I compared it from the IRIS dataset available on Kaggle.

•	Then there is the problem of having to edit the CSV files not to have anything other than the dataset itself.

•	I also have to predefine the number of data points and attributes even though they can be taken from the file itself, but I need to allocate the memory first in the driver code.

•	I also could not plot the points to represent the clusters.
