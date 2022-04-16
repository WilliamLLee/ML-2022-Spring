# implements the linear discriminant analysis with probit regression algorithm
# Author: Li Wei
# Date: 2022/04/12
# this :module :module `LDA_PR.py`  provide a class of LDA_PR

import numpy as np
from utls import *

class LDA_PR:
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

    def fit(self,X,y,lr = 0.01,max_epoch= 1000,batch_size = 100,epsilon = 1e-6,show_parameter= False):
        '''
        :param X: the input data
        :param y: the output data, represented as one-hot type
        :return: the model
        '''
        X = np.array(X)
        y = np.array(y)
        x_N, _ = X.shape
        all_loss = []
        for epoch in range(1,max_epoch+1):
            shuffle_index = np.random.permutation(x_N)
            X, y = X[shuffle_index],y[shuffle_index]
            loss = 0.0
            for iter in range(0,x_N,batch_size):
                X_batch = X[iter:iter+batch_size,:]
                y_batch = y[iter:iter+batch_size,:]
                prob = self.predict(X_batch)
                # cross entroy loss for probit regression
                loss += - np.sum(np.sum(y_batch*np.log(prob)+(1-y_batch)*np.log(1-prob),axis=1))
                grad_w = np.dot(X_batch.T,prob-y_batch)
                grad_b = np.sum(prob-y_batch,axis=0)
                self._w -= lr*grad_w
                self._b -= lr*grad_b
            loss = loss/x_N
            
            if show_parameter and epoch%1000 == 0:
                print('epoch:',epoch,'loss:',loss)
                if len(all_loss) >= 1 and abs(all_loss[-1] - loss) < epsilon:
                    break

            all_loss.append(loss)

        if show_parameter:
            print('weight:',self._w)
            print('bias:',self._b)
        return all_loss

    def predict(self,X):
        X = np.array(X)
        prob = np.dot(X,self._w)+self._b  # linear layer
        prob = probit(prob)               # probit activation function
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