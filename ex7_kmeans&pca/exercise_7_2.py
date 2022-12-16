import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

'''
2.  Implementing Principal Component Analysis
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
    plt.scatter(x=x1, y=x2, color='b', marker='o')
    plt.show()

# normalize the dataset
def feature_normalized(X):
    X_normalized = np.matrix(np.zeros(X.shape))
    for i in range(X.shape[1]):
        X_normalized[:, i] = np.matrix((X[:, i] - X[:, i].mean()) / X[:, i].std()).T
    return X_normalized

# PCA algorithm
def pca(X, K):
    Sigma = (1 / X.shape[0]) * (X.T * X)    # (2,50)*(50,2)=(2,2)
    U, S, V = np.linalg.svd(Sigma)
    U_reduce = U[:, :K] # (2,1)
    # Z = np.matrix(np.zeros((X.shape[0], K)))    # (50,1) new features---primary components
    Z = (U_reduce.T * X.T).T  # (1,2).*(2,50)=(1,50)
    return Z, U_reduce

# reconstruct the dataset
def reconstruct_data(Z, U_reduce):
    X_approximate = (U_reduce * Z.T).T    # (2,1)*(1,50)=(2,50)
    return X_approximate

# draw connecting lines
def draw_connections(X_normalized, X_approximate):
    plt.figure()
    plt.scatter(x=X_normalized[:, 0].flatten().A[0], y=X_normalized[:, 1].flatten().A[0], marker='o', facecolors='none', edgecolors='b')
    plt.scatter(x=X_approximate[:, 0].flatten().A[0], y=X_approximate[:, 1].flatten().A[0], marker='o', facecolors='none', edgecolors='r')
    for i in range(X_normalized.shape[0]):
        x1 = X_normalized[i, 0]
        y1 = X_normalized[i, 1]
        x2 = X_approximate[i, 0]
        y2 = X_approximate[i, 1]
        plt.plot([x1, x2], [y1, y2], color='k', linestyle='--')
    plt.axis([-4, 3, -4, 3])
    plt.show()

if __name__ == '__main__':
    # read the dataset
    path1 = 'D:/新建文件夹/机器学习/Machine_Learning_exercise/exercise_7/ex7/ex7data1.mat'
    X1 = read_data(path1)
    # visualize the dataset
    visualize_data(X1)
    # data preprocessing
    # normalization
    X1_normalized = feature_normalized(X1)
    # run the PCA algorithm
    K1 = 1
    Z1, U1_reduce = pca(X1_normalized, K1)
    print('projections of the original features on the primary components are \n', Z1)
    # reconstruct the approximation of the data
    X1_approximate = reconstruct_data(Z1, U1_reduce)
    print('approximations of the new features are: \n', X1_approximate)
    # draw lines connecting the projections to the original data points
    draw_connections(X1_normalized, X1_approximate)
    '''Due to time is limited, the next task of the face image is put aside temporarily'''
