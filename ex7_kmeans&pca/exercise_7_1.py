import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import cv2
from sklearn.cluster import KMeans

'''
1.  Implementing K-means
'''

# read the dataset
def read_data(path):
    data = loadmat(path)
    X = data['X']
    return X

# visualize the dataset
def visualize_data(X):
    x1 = X[:, 0]
    x2 = X[:, 1]
    plt.figure()
    plt.scatter(x=x1, y=x2, color='k', marker='o')
    plt.show()

# find the closest centroids for data
def find_closest_centroids(X, centroids, K):
    X_index = np.matrix(np.zeros((X.shape[0], 1)))
    distance = np.matrix(np.zeros((X.shape[0], 1)))
    distance_temp = np.matrix(np.zeros((K, 1)))
    for m in range(X.shape[0]):
        for k in range(K):
            distance_temp[k, 0] = np.power(np.sum(np.power((X[m, :] - centroids[k, :]), 2)), 0.5)   # compute the distance between data points and present cnetroids
        X_index[m, 0] = np.argmin(distance_temp)
        distance[m, 0] = distance_temp[int(X_index[m, 0]), 0]
    return X_index

# compute the new centroids
def compute_centroids(X, X_index, K):
    new_centroids = np.matrix(np.zeros((K, X.shape[1])))
    for k in range(K):
        X_k = np.matrix(X[np.argwhere(X_index == k)[:, 0], :])
        new_centroids[k, :] = np.sum(X_k, axis=0) / X_k.shape[0]
    return new_centroids

# k-means algorithm
def k_means(X, K, initial_centroids, max_iter):
    present_centroids = initial_centroids
    new_centroids = initial_centroids
    for i in range(max_iter):
        # find the closest centroids for data
        X_index = find_closest_centroids(X, present_centroids, K)
        # compute the new centroids
        new_centroids = compute_centroids(X, X_index, K)
        present_centroids = new_centroids
    return new_centroids, X_index

# using sklearn.cluster.KMeans
def sklearn_cluster(X, K):
    model_original = KMeans(n_clusters=K)
    print('the KMeas model is ', model_original)
    model = model_original.fit(X)
    centroids = model_original.cluster_centers_
    print('the centroids computed by sklearn.cluster.KMeans are \n', centroids)
    X_index = model_original.predict(X)
    visualize_cluster(centroids, X_index, X)

# visualize the new clusters
def visualize_cluster(centroids, X_index, X):
    centroids = np.matrix(centroids)
    plt.figure()
    color = np.matrix(['r', 'g', 'b'])
    for k in range(centroids.shape[0]):
        X_k = np.matrix(X[np.argwhere(X_index == k)[:, 0], :])
        x1 = X_k[:, 0].flatten().A[0]
        x2 = X_k[:, 1].flatten().A[0]
        plt.scatter(x=x1, y=x2, color=color[0, k], marker='o')
    plt.scatter(x=centroids[:, 0].flatten().A[0], y=centroids[:, 1].flatten().A[0], color='k', marker='+')
    plt.show()

# initialize the centroids randomly---choosing the data points as the initial centroids
def initialize_centroids(X, K):
    initial_centroids_index = np.random.randint(0, X.shape[0], K)
    initial_centroids_image = X[initial_centroids_index[:], :]
    return initial_centroids_image

if __name__ == '__main__':
    '''K-Means'''
    # read the dataset
    path1 = 'D:/新建文件夹/机器学习/Machine_Learning_exercise/exercise_7/ex7/ex7data2.mat'
    X1 = read_data(path1)
    # visualize the dataset
    visualize_data(X1)
    # initialize parameters
    K = 3
    initial_centroids = np.matrix([[3, 3], [6, 2], [8, 5]])
    # run the k-means algorithm
    max_iter = 10
    new_centroids, X_index = k_means(X1, K, initial_centroids, max_iter)
    print('the new cluster centroids are \n', new_centroids)
    # visualize the new clusters
    visualize_cluster(new_centroids, X_index, X1)
    # using the algorithm in sklearn.cluster.KMeans
    sklearn_cluster(X1, K)
    '''K-Means Clustering on Pixels'''
    # read the picture
    path2 = 'bird_small.png'
    image1 = cv2.imread(path2)
    # convert BGR to RGB and show the picture
    image1_RGB = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    plt.figure()
    plt.imshow(image1_RGB)
    plt.show()
    # read the pixel data of the image
    path3 = 'D:/新建文件夹/机器学习/Machine_Learning_exercise/exercise_7/ex7/bird_small.mat'
    data_image1 = loadmat(path3)
    A = data_image1['A']
    A = A / 255
    X_image = np.reshape(A, ((A.shape[0] * A.shape[1]), 3))
    # initialize parameters
    K_image = 16
    # initialize the centroids randomly
    initial_centroids_image = initialize_centroids(X_image, K_image)
    # run the k-means algorithm
    max_iter = 10
    new_centroids_image, X_image_index = k_means(X_image, K_image, initial_centroids_image, max_iter)
    print('the primary colours of the image are \n', new_centroids_image)
    # visualize the new picture with the primary colours
    X_new_image = new_centroids_image[X_image_index.flatten().A[0].astype(int), :]
    X_new_image_ndarray = X_new_image.A
    X_new_image_show = np.reshape(X_new_image_ndarray, (A.shape[0], A.shape[1], A.shape[2]))
    plt.figure()
    plt.imshow(X_new_image_show)
    plt.show()
