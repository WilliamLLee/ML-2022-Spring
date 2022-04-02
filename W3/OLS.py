#!/user/bin/env python
# -*- coding:utf-8 -*-
# author: Wei Li
# creat: 2022-3-29
# modify: 2022-3-29
# function: Implement the ordinary least square (OLS) method. 

import numpy as np

from utls import square_error

class OLS(object):
    '''
    Fit the input data samples with the OLS method.
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
        return (np.array(x)).dot(self._weight)+self._bias

    def square_error(self,x,y):
        x = np.array(x)
        y = np.array(y)
        y_pred = self.predict(x)
        return square_error(y_pred,y)

    def fit(self,x,y,show_parameters=True):
        x = np.array(x)
        y = np.array(y)
        x = np.hstack((x,np.ones((x.shape[0],1))))  # add the bias
        omiga = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)  # solve the equation
        self._weight = omiga[:-1,:]
        self._bias =omiga[-1:,:]
        if show_parameters:
            print("[OLS] weight matrix:\n",self._weight)
            print("[OLS] bias vector:\n",self._bias)



