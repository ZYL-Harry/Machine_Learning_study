import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

'''
1. Anomaly Detection
'''

# read the dataset
def read_data(path):
    data = loadmat(path)
    X = data['X']
    Xval = data['Xval']
    yval = data['yval']
    return X, Xval, yval

# visualize the dataset
def visualize_data(X):
    plt.figure()
    x1 = X[:, 0]
    x2 = X[:, 1]
    plt.scatter(x=x1, y=x2, color='b', marker='+')
    plt.xlabel('Latency(ms)')
    plt.ylabel('Throughput(mb/s)')
    plt.show()

# estimate parameters for the Gaussian distribution for each features
def estimate_parameters(X):
    mu = np.matrix(np.zeros((X.shape[1], 1)))
    sigma2 = np.matrix(np.zeros((X.shape[1], 1)))
    for j in range(X.shape[1]):
        mu[j, 0] = X[:, j].mean()
        sigma2[j, 0] = X[:, j].var()
    return mu, sigma2

# compute the probability with the Gaussian distribution and plot it
def estimate_probability(X, mu, sigma2):
    p_j = np.matrix(np.zeros((X.shape[0], 2)))
    p = np.matrix(np.ones((X.shape[0], 1)))   # initialize the p for multiply
    for j in range(X.shape[1]):
        p_j[:, j] = np.matrix((1 / (np.power((2 * np.pi), 0.5) * np.power(sigma2[j, 0], 0.5))) * np.exp(- (np.power((X[:, j] - mu[j, 0]), 2)) / (2 * sigma2[j, 0]))).T
        p = np.multiply(p, p_j[:, j])
    return p

# visualize the probability
def visualize_probability(X, mu, sigma2):
    X1 = np.linspace(start=0, stop=30, num=100)
    X2 = np.linspace(start=0, stop=30, num=100)
    X1_plot, X2_plot = np.meshgrid(X1, X2)
    X1_plot = np.matrix(X1_plot)
    X2_plot = np.matrix(X2_plot)
    X1_estimate = np.reshape(X1_plot, ((X1_plot.shape[0] * X1_plot.shape[1]), 1))
    X2_estimate = np.reshape(X2_plot, ((X2_plot.shape[0] * X2_plot.shape[1]), 1))
    X_estimate = np.c_[X1_estimate, X2_estimate]
    p = estimate_probability(X_estimate.A, mu, sigma2)
    p_plot = np.reshape(p, X1_plot.shape)
    plt.figure()
    plt.scatter(x=X[:, 0], y=X[:, 1], color='b', marker='+')
    contour_levels = [10 ** h for h in range(-20, 0, 3)]
    plt.contour(X1_plot, X2_plot, p_plot, contour_levels)
    plt.xlabel('Latency(ms)')
    plt.ylabel('Throughput(mb/s)')
    plt.show()

# select the threshold
def select_threshold(yval, pval):
    threshold_set = np.linspace(start=pval.min(), stop=pval.max(), num=1000)
    threshold_best = 0
    F1_best = 0
    for i in range(threshold_set.shape[0]):
        tp = 0
        fp = 0
        fn = 0
        threshold = threshold_set[i]
        # for j in range(yval.shape[0]):
        #     if (yval[j, 0] == 1) & (pval[j, 0] < threshold):
        #         tp = tp + 1
        #     elif (yval[j, 0] == 0) & (pval[j, 0] < threshold):
        #         fp = fp + 1
        #     elif (yval[j, 0] == 1) & (pval[j, 0] > threshold):
        #         fn = fn + 1
        tp = np.sum(np.logical_and((yval == 1), (pval < threshold))).astype(int)
        fp = np.sum(np.logical_and((yval == 0), (pval < threshold))).astype(int)
        fn = np.sum(np.logical_and((yval == 1), (pval > threshold))).astype(int)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        F1 = (2 * precision * recall) / (precision + recall)
        if F1 > F1_best:
            threshold_best = threshold
            F1_best = F1
    return threshold_best, F1_best

# pick out the anomalies and plot them
def pick_anomalies(X, p, threshold, mu, sigma2):
    # plot the original data and contour
    X1 = np.linspace(start=0, stop=30, num=100)
    X2 = np.linspace(start=0, stop=30, num=100)
    X1_plot, X2_plot = np.meshgrid(X1, X2)
    X1_plot = np.matrix(X1_plot)
    X2_plot = np.matrix(X2_plot)
    X1_estimate = np.reshape(X1_plot, ((X1_plot.shape[0] * X1_plot.shape[1]), 1))
    X2_estimate = np.reshape(X2_plot, ((X2_plot.shape[0] * X2_plot.shape[1]), 1))
    X_estimate = np.c_[X1_estimate, X2_estimate]
    p_estimate = estimate_probability(X_estimate.A, mu, sigma2)
    p_plot = np.reshape(p_estimate, X1_plot.shape)
    plt.figure()
    plt.scatter(x=X[:, 0], y=X[:, 1], color='b', marker='+')
    contour_levels = [10 ** h for h in range(-20, 0, 3)]
    plt.contour(X1_plot, X2_plot, p_plot, contour_levels)
    # pick out the anomalies
    anomaly_index = np.matrix(np.argwhere(p < threshold))
    x1_anomaly = X[anomaly_index[:, 0], 0]
    x2_anomaly = X[anomaly_index[:, 0], 1]
    plt.scatter(x=x1_anomaly, y=x2_anomaly, marker='o', facecolors='none', edgecolors='r', linewidths=2)
    plt.xlabel('Latency(ms)')
    plt.ylabel('Throughput(mb/s)')
    plt.show()

if __name__ == '__main__':
    # read the dataset
    path1 = 'D:/新建文件夹/机器学习/Machine_Learning_exercise/exercise_8/ex8/ex8data1.mat'
    X, Xval, yval = read_data(path1)
    # visualize the dataset
    visualize_data(X)
    # estimate parameters for the Gaussian distribution for each features
    mu, sigma2 = estimate_parameters(X)
    print('μ = ', mu.flatten().A[0])
    print('σ^2 = ', sigma2.flatten().A[0])
    # compute the probability with Gaussian distribution
    p = estimate_probability(X, mu, sigma2)
    # visualize the probability
    visualize_probability(X, mu, sigma2)
    # select the threshold with the cross validation set
    pval = estimate_probability(Xval, mu, sigma2)
    threshold, F1 = select_threshold(yval, pval)
    print('best threshold found with the cross validation set is ', threshold)
    print('F1 of the best threshold on the cross validation set is', F1)
    # pick out the anomalies
    pick_anomalies(X, p, threshold, mu, sigma2)
    '''high dimensional dataset'''
    path2 = 'D:/新建文件夹/机器学习/Machine_Learning_exercise/exercise_8/ex8/ex8data2.mat'
    X2, Xval2, yval2 = read_data(path2)
    mu_2, sigma2_2 = estimate_parameters(X2)
    p_2 = estimate_probability(X2, mu_2, sigma2_2)
    pval_2 = estimate_probability(Xval2, mu_2, sigma2_2)
    threshold_2, F1_2 = select_threshold(yval2, pval_2)
    print('the best threshold found with the cross validation is ', threshold_2)
    anomaly_index_2 = np.matrix(np.argwhere(p_2 < threshold_2))
    print('there are {0} anomalies'.format(int(anomaly_index_2.shape[0])))
