import numpy as np
import matplotlib.pyplot as plt 

def f(x):
    '''
        The function f(x)
        @param: x, the x-axis
        @return: the value of f(x)
    '''
    return -np.sin(x/5)+np.cos(x)

def poly(x, n):
    '''
        Calculate the polynomial
        @param: x, the x-axis
                n, the degree of the polynomial
        @return: the value of the polynomial
    '''
    x_poly = np.ones(x.shape[0]).reshape(-1,1)
    for i in range (1,n+1):
        x_poly = np.hstack((x_poly, x**(i)))
    return x_poly[:,1:]

def square_error(y_pred,y_true):
    '''
        Calculate the square loss
        @param: y_pred, the predicted value
                y_true, the true value
        @return: the square error
    '''
    input_size_1 = y_pred.shape[0]
    input_size_2 = y_true.shape[0]
    
    if input_size_1 != input_size_2:
        raise ValueError("The size of y_pred and y_true must be the same!")
    if y_pred.shape[1] != y_true.shape[1]:
        raise ValueError("The size of y_pred and y_true must be the same!")
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    return np.sum(np.sum(((y_true - y_pred) ** 2), axis=1) / y_pred.shape[1]) / y_pred.shape[0]

def Generate_Samples(N,X_range,mu,sigma,func):
    '''
        Generate the samples with the function f(x) and error epsilons
        @param: N the, samples numbers 
                X_range, the value range of x_i
                mu, the mean of the error normal distribution
                sigma, the standard deviation of the error normal distribution
        @return: samples, the generated samples. 
    '''
    # Ensure consistent generation of random numbers to facilitate debugging and display results, you can comment out
    np.random.seed(1)  
    # Generate samples
    epsilons = np.random.normal(mu,sigma,N)
    X = np.random.uniform(X_range[0],X_range[1],N)
    samples = []
    for x_i, epsilon in zip(X,epsilons):
        samples.append((x_i,func(x_i)+epsilon))
    return samples,epsilons

def plot_samples(samples,title=None,show = False):
    '''
        Plot the samples
        @param: samples, the samples to be plotted
                title, the title of the plot
    '''
    x = [x_i for x_i,_ in samples]
    y = [y_i for _,y_i in samples]
    plt.plot(x,y,'x')  
    if title is not None:
        plt.title(title)
    if show:
        plt.show()

def plot_samples_with_error(samples,epsilons,title=None,show = False):
    '''
        Plot the samples with error
        @param: samples, the samples to be plotted
                epsilons, the error of the samples
                title, the title of the plot
    '''
    x = [x_i for x_i,_ in samples]
    y = [y_i for _,y_i in samples]
    plt.errorbar(x,y,yerr=epsilons,fmt='x')
    if title is not None:
        plt.title(title)
    if show:
        plt.show()

def draw_lines(x,y,title=None,show = False,with_marker = False):
    '''
        Draw the lines
        @param: x, the x-axis
                y, the y-axis
                title, the title of the plot
    '''
    if with_marker:
        plt.plot(x,y,color='red',linewidth=1,marker = 'x',markeredgecolor='blue')
    else:
        plt.plot(x,y,color='red',linewidth=1)
    if title is not None:
        plt.title(title)
    if show:
        plt.show()

def draw_histogram(x,x_range,title=None,show = False):
    '''
        Draw the histogram
        @param: x, the x-axis
                x_range, the value range of x_i
                title, the title of the plot
    '''
    plt.hist(x,x_range)
    if title is not None:
        plt.title(title)
    if show:
        plt.show()


if __name__=="__main__":
    x =np.array([[1],[2],[3],[4],[5]])
    n = 2
    print(poly(x,n))