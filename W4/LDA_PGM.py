# implements the probabilistic linear discriminant analysis with generative model
# author: Li Wei
# date: 2022/04/12
# this :module `LDA_PGM.py`  provide a class of LDA_PGM

import numpy as np
from utls import sigmoid

class LDA_PGM:
    def __init__(self, input_dim,output_dim):
        '''
        :param input_dim: the input dimension of the data
        :param output_dim: the output dimension of the data
        :param save_model: whether to save the model
        '''
        self.input_dim = input_dim
        self.output_dim = output_dim
        self._w = np.zeros((self.input_dim,self.output_dim))
        self._b = np.zeros((1,self.output_dim))


    def fit(self,X,y):
        '''
        :param X: the input data
        :param y: the output data, two class and the output_dim is 1, with the value 0 or 1
        :return: the model
        '''
        X = np.array(X)
        y = np.array(y)  
        
        # get the mean of each class
        mean_y1 = np.mean(X[y.flatten()==1],axis=0)
        mean_y2 = np.mean(X[y.flatten()==0],axis=0)
        # print(mean_y1)
        # print(mean_y2)

        # get the covariance matrix of class 1
        cov = np.cov(X.T)
        print(cov.shape)
        # get the prior probability of each class
        prior_y1 = np.sum(y.flatten()==1)/len(y)
        prior_y2 = np.sum(y.flatten()==0)/len(y)

        # get the weight matrix
        inverse_cov_y1 = np.linalg.pinv(cov)
        

        self._w = np.dot(inverse_cov_y1,mean_y1-mean_y2).reshape(-1)
        self._b = -0.5*np.dot(np.dot(mean_y1.T,inverse_cov_y1),mean_y1)+0.5*np.dot(np.dot(mean_y2.T,inverse_cov_y1),mean_y2)+np.log(prior_y1/prior_y2)
       

    def predict(self,X):
        X = np.array(X)
        prob = np.dot(X,self._w)+self._b
        prob = sigmoid(prob)
        return prob
    
    def get_w(self):
        return self._w
    
    def get_b(self):
        return self._b
    
    def save_model(self,save_path):
        np.save(save_path+self.__class__.__name__,{
            'wight':self._w,
            'bias':self._b
        })
