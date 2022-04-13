# implements the linear discriminant analysis with OLS algorithm
# Author: Li Wei
# Date: 2022/04/12
# this :module `LDA_OLS.py`  provide a class of LDA_OLS


import numpy as np

class LDA_OLS:
    def __init__(self,input_dim,output_dim):
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
        :param y: the output data, represented as one-hot type
        :return: the model
        '''
        X = np.array(X)
        y = np.array(y)
        X = np.hstack((X,np.ones((X.shape[0],1))))  
        temp_weight = np.dot(np.linalg.pinv(X),y)
        self._w = temp_weight[:-1,:]
        self._b = temp_weight[-1:,:]

    def predict(self,X):
        X = np.array(X)
        prob = np.dot(X,self._w)+self._b
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

    def load_model(self,load_path):
        model = np.load(load_path+self.__class__.__name__+'.npy',allow_pickle=True)
        self._w = model['wight']
        self._b = model['bias']
        

        
        
        