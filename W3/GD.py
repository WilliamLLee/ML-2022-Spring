#!/user/bin/env python
# -*- coding:utf-8 -*-
# author: Wei Li
# creat: 2022-3-29
# modify: 2022-3-29
# function: Implement the gradient descent (GD) method.

import numpy as np

from utls import square_error

class GD(object):
    '''
    Fit the input data samples with the GD method.
    @param: input_dim, the input dimension
            output_dim, the output dimension
            x , the input data
            y , the output data
    '''

    def __init__(self,input_dim,output_dim) -> None:
        self._input_dim = input_dim
        self._output_dim = output_dim 
        self._weight = np.mat(np.zeros((self._input_dim,self._output_dim)))
        self._bias = np.mat(np.zeros((1,self._output_dim)))
    
    def predict(self,x):
        return np.array(x).dot(self._weight)+self._bias
    
    def square_error(self,x,y):
        x = np.array(x)
        y = np.array(y)
        y_pred = self.predict(x)
        return square_error(y_pred,y)

    def fit(self,x,y,learning_rate=0.01,epochs=1000,show_parameters=True):
        x = np.mat(x)
        y = np.mat(y)
        total_loss = []
        x_N,_ = x.shape
        for epoch in range(epochs):
            pred_y = self.predict(x).reshape(-1,1)
            loss = square_error(pred_y,y)
            dW = np.mat(np.zeros((self._input_dim,self._output_dim)))
            db = np.mat(np.zeros((1,self._output_dim)))
            for i in range(x_N):
                dW += x[i,:]*(pred_y[i]-y[i])
                db += pred_y[i,:]-y[i,:]
            # update the weight and bias
            self._weight = self._weight - learning_rate*dW
            self._bias = self._bias - learning_rate*db
            
            if epoch%10 == 0:
                print("[GD] epoch: ",epoch," loss: ",loss)
                if len(total_loss)>=1 and total_loss[-1] - loss < 0.00001:
                    print("[GD] converge at epoch: ",epoch)
                    break 
            total_loss.append(loss)
            
        print("[GD] stop training.")
        if show_parameters:
            print("[GD] weight matrix:\n ",self._weight)
            print("[GD] bias vector:\n ",self._bias)