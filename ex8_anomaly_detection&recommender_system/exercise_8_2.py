import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.optimize as opt

'''
2. Recommender Systems
'''

# read the dataset---ratings
def read_ratings(path):
    data = loadmat(path)
    R = data['R']
    Y = data['Y']
    return R, Y

# read the dataset---parameters
def read_parameters(path):
    data = loadmat(path)
    num_features = int(data['num_features'])
    num_movies = int(data['num_movies'])
    num_users = int(data['num_users'])
    Theta = data['Theta']
    X = data['X']
    return num_features, num_movies, num_users, Theta, X

# learn the density of the data collection
def collection_density(R):
    plt.figure()
    plt.imshow(R)
    plt.xlabel('Users')
    plt.ylabel('Movies')
    plt.show()

# vectorization
def vectorization(X, Theta):
    # print('X', X.flatten().shape)
    X_Theta = np.r_[X.flatten().T, Theta.flatten().T]
    # print('X_Theta', X_Theta.shape)
    return X_Theta

# recover the shape from vectorization
def recover_vectorization(X_Theta, num_movies, num_users, num_features):
    X = np.matrix(np.reshape(X_Theta[:(num_movies * num_features)], (num_movies, num_features)))
    Theta = np.matrix(np.reshape(X_Theta[(num_movies * num_features):], (num_users, num_features)))
    return X, Theta

# cost function
def cost_function(X_Theta, Y, R, learning_rate, num_movies, num_users, num_features):
    X, Theta = recover_vectorization(X_Theta, num_movies, num_users, num_features)
    Y = np.matrix(Y)
    R = np.matrix(R)
    error = X * Theta.T - Y   # (1682,10)*(10,943)=(1682,943)
    regularized_part_X = (learning_rate / 2) * np.sum(np.power(Theta, 2))
    regularized_part_Theta = (learning_rate / 2) * np.sum(np.power(X, 2))
    costfunction = (1 / 2) * np.sum(np.power((np.multiply(error, R)), 2)) + regularized_part_X + regularized_part_Theta
    return costfunction

# gradient descent
def gradient_descent(X_Theta, Y, R, learning_rate, num_movies, num_users, num_features):
    X, Theta = recover_vectorization(X_Theta, num_movies, num_users, num_features)
    Y = np.matrix(Y)
    R = np.matrix(R)
    error = X * Theta.T - Y # (1682,943)
    X_gradient = np.multiply(error, R) * Theta + learning_rate * X # (1682,10)
    Theta_gradient = np.multiply(error, R).T * X + learning_rate * Theta  # (943,10)
    X_Theta_gradient = vectorization(X_gradient.A, Theta_gradient.A)
    return X_Theta_gradient

# read the movie list and enter preferences
def read_rating(Y, R):
    path_movie = 'movie_ids.txt'
    movie_data = open(path_movie, 'r', encoding='latin 1')
    movie_list = []
    for line in movie_data:
        tokens = line.strip().split(' ')
        movie_list.append(' '.join(tokens[1:]))
    movie_list = np.array(movie_list)
    # print(movie_list)
    new_ratings = np.matrix(np.zeros((Y.shape[0], 1)))
    new_ratings[0, 0] = 4
    new_ratings[6, 0] = 3
    new_ratings[11, 0] = 5
    new_ratings[53, 0] = 4
    new_ratings[63, 0] = 5
    new_ratings[65, 0] = 3
    new_ratings[68, 0] = 5
    new_ratings[97, 0] = 2
    new_ratings[182, 0] = 4
    new_ratings[225, 0] = 5
    new_ratings[354, 0] = 5
    Y = np.c_[Y, new_ratings]
    R = np.c_[R, (new_ratings != 0)]
    return movie_list, Y, R

# do the mean normalization
def mean_normalization(Y):
    Y_mean = np.matrix(np.zeros((Y.shape[0], 1)))
    Y_normalized = np.matrix(np.zeros(Y.shape))
    for i in range(Y.shape[0]):
        index = np.argwhere(R[i, :] == 1)
        Y_mean[i, 0] = Y[i, index].mean()
        Y_normalized[i, index] = np.matrix(Y[i, index] - Y_mean[i, 0])
    return Y_mean, Y_normalized

if __name__ == '__main__':
    # read the ratings and parameters
    path1 = 'D:/新建文件夹/机器学习/Machine_Learning_exercise/exercise_8/ex8/ex8_movies.mat'
    R, Y = read_ratings(path1)
    path2 = 'D:/新建文件夹/机器学习/Machine_Learning_exercise/exercise_8/ex8/ex8_movieParams.mat'
    num_features, num_movies, num_users, Theta, X = read_parameters(path2)
    # learn the density of the data collection
    collection_density(R)
    # collaborative filtering learning algorithm
    learning_rate = 10
    '''test'''
    # costfunction = cost_function(Theta, X, Y, R, learning_rate)
    '''practical'''
    # read the movie list and enter preferences
    movie_list, Y_new, R_new = read_rating(Y, R)
    # do the mean normalization
    Y_mean, Y_normalized = mean_normalization(Y_new)
    # find the best parameters
    X_initial = np.random.random((num_movies, num_features))
    Theta_initial = np.random.random((num_users, num_features))
    X_Theta = vectorization(X_initial, Theta_initial)
    result = opt.minimize(fun=cost_function, x0=X_Theta, args=(Y, R, learning_rate, num_movies, num_users, num_features), method='CG', jac=gradient_descent, options={'maxiter': 100})
    X_Theta_best = result.x
    X_best, Theta_best = recover_vectorization(X_Theta_best, num_movies, num_users, num_features)
    # predict the unknown ratings
    prediction = X_best * Theta_best.T
    real_prediction = prediction + Y_mean
    # sort the prediction and find the top 10
    new_user = real_prediction[:, (real_prediction.shape[1] - 1)]
    recommend_index = np.argsort(- new_user, axis=0)
    top_10 = recommend_index[:10]
    print('Top recommendations: ')
    t = 0
    for i in top_10:
        t = t + 1
        print('{0}: {1} {2}'.format(t, float(new_user[i]), movie_list[i][0][0]))



