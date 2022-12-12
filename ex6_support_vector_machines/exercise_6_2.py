import numpy as np
from scipy.io import loadmat
from sklearn import svm
import pandas as pd

'''
2. Spam Classification
'''

# train the svm algorithm
def training_data(X, y):
    model = svm.SVC(C=100, kernel='linear')
    estimator = model.fit(X, y)
    accuracy = model.score(X, y)
    print('an instance of estimator is ', estimator)
    print('the mean accuracy is ', accuracy)
    return model

if __name__ == '__main__':
    # read the dataset
    path1 = 'D:/新建文件夹/机器学习/Machine_Learning_exercise/exercise_6/ex6/spamTrain.mat'
    data1 = loadmat(path1)
    X = data1['X']
    y = data1['y']
    # train svm algorithm
    model = training_data(X, y)
    # compute the mean accuracy of the test dataset
    path2 = 'D:/新建文件夹/机器学习/Machine_Learning_exercise/exercise_6/ex6/spamTest.mat'
    data2 = loadmat(path2)
    Xtest = data2['Xtest']
    ytest = data2['ytest']
    test_accuracy = model.score(Xtest, ytest)
    print('the mean accuracy of the test dataset is ', test_accuracy)
    # see which words the classifier thinks are the most predictive of spam
    # find the index and the order of the words
    w = np.matrix(model.coef_).T
    w_sort = np.sort(-w, axis=0)    # sort from largest to smallest
    w_index = np.argsort(-w, axis=0)
    w_new = np.c_[w_index + 1, w_sort]
    # print(w_new[:15, :])
    # get volcabulary list
    path_volcabulary = 'D:/新建文件夹/机器学习/Machine_Learning_exercise/exercise_6/ex6/vocab.txt'
    data_volcabulary = pd.read_csv(path_volcabulary, header=None, names=['index', 'volcabulary'], sep='\t')
    # print(data_volcabulary.head())
    index = np.matrix(data_volcabulary.loc[:, ['index']].values)
    volcabulary = np.matrix(data_volcabulary.loc[:, ['volcabulary']].values)
    # find the spam words
    spam_words = volcabulary[w_index[0:15, 0], 0]
    print(spam_words)

    # another method that hasn't been completed
    # word_index = np.eye(X.shape[1])
    # word_predict = np.matrix(model.decision_function(word_index)).T
    # word_spam = word_predict(np.argwhere(word_predict > 0.55))
