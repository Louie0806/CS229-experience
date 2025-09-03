from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np


A = imread("peppers-large.tiff")
B = imread("peppers-small.tiff")
if A.dtype == np.uint8:
    A = A.astype(float) / 255.0
if B.dtype == np.uint8:
    B = B.astype(float) / 255.0

k = 16


def kmeans(data, k, max_iter=30):

    centroids = data[np.random.choice(data.shape[0], k, replace=False)]

    for it in range(max_iter):

        distances = np.sum((data[:, np.newaxis, :] - centroids) ** 2, axis=2)
        labels = np.argmin(distances, axis=1)

        new_centroids = np.zeros_like(centroids)
        for i in range(k):
            cluster_points = data[labels == i]
            if len(cluster_points) > 0:
                new_centroids[i] = cluster_points.mean(axis=0)
            else:

                new_centroids[i] = data[np.random.randint(data.shape[0])]

        if np.allclose(centroids, new_centroids, atol=1e-5):
            break
        centroids = new_centroids

    return centroids


B_flat = B.reshape(-1, 3)
centroids = kmeans(B_flat, k)

A_flat = A.reshape(-1, 3)
distances = np.sum((A_flat[:, np.newaxis, :] - centroids) ** 2, axis=2)
labels = np.argmin(distances, axis=1)

C = centroids[labels].reshape(A.shape)

plt.imshow(C)
plt.show()
