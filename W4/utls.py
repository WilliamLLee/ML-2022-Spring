# provides utility functions for the project
# Author: Li Wei
# Date: 2022/04/12
# this :module `utls.py`  provide some utility functions

import numpy as np
from regex import P
import scipy.stats as st

def one_hot(labels):
    '''
    :param labels: the labels of the data, as a shape of (n,1)
    :return: the labels as one-hot type
    '''
    # count the number of labels type
    label_count = len(set(np.array(labels).flatten()))
    # initialize the one-hot type
    one_hot_labels = np.zeros((len(labels),label_count))
    # translate the labels as one-hot type
    for i in range(len(labels)):
        one_hot_labels[i][labels[i]] = 1
    return one_hot_labels

def normalize(x):
    '''
    :param x: the input
    :return: the normalized input
    '''
    return (x-np.min(x))/(np.max(x)-np.min(x))

def test(model,test_features,test_labels):
    '''
    :param model: the model
    :param test_features: the test features
    :param test_labels: the test labels
    :return: the error of the model
    '''
    np.array(test_features)
    np.array(test_labels)

    # get the predict labels probability 
    predict_labels = model.predict(test_features)  # (n,1)
    # print("+",predict_labels.flatten()[:30])
    predict_labels = np.int64(predict_labels>0.5).flatten()
    # print("-",predict_labels[:30],test_labels.flatten()[:30])
    # get the error
    error = np.sum(np.array(predict_labels) != np.array(test_labels.flatten()))/len(test_labels.flatten())
    return error

def sigmoid(inx):
    '''
    :param x: the input, as a shape of (n,1)
    :return: the sigmoid of the input
    '''
    # prevent overflow
    ans = np.zeros(inx.shape)

    ans[inx>=0] = 1/1+np.exp(-inx[inx>=0])
    ans[inx<0] = np.exp(inx[inx<0])/(1+np.exp(inx[inx<0]))
    return ans

def softmax(x):
    '''
    :param x: the input
    :return: the softmax function return of the input
    '''
    return np.exp(x)/np.sum(np.exp(x))

def probit(x):
    '''
    :param x: the input
    :return: the probit function return of the input
    '''
    return st.norm.cdf(x)

if __name__ == '__main__':
    # test functions
    predict_labels = np.array([[0.1,0.9],[0.8,0.2]])
    # get the predict labels
    predict_labels = np.argmax(predict_labels,axis=1)
    test_labels = np.array([1,0])
    print(np.sum(np.array(predict_labels) != np.array(test_labels)))