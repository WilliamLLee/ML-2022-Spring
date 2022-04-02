import numpy as np

def f(x):
    return np.sin(x/5)+np.cos(x)

def Generate_Samples(N,X_range,mu,sigma,func):
    '''
        Generate the samples with the function f(x) and error epsilons
        @param: N the, samples numbers 
                X_range, the value range of x_i
                mu, the mean of the error normal distribution
                sigma, the standard deviation of the error normal distribution
        @return: samples, the generated samples. 
    '''
    epsilons = np.random.normal(mu,sigma,N)
    X = np.random.uniform(X_range[0],X_range[1],N)
    samples = []
    for x_i, epsilon in zip(X,epsilons):
        samples.append((x_i,func(x_i)+epsilon))
    return samples