import numpy as np
import matplotlib.pyplot as plt

def regPlot(X,Y,Ypred,slope=1,intercept=0):
    """ plot the result including rediuals. 
    Parameters:
    X - x-values (numpy array) 
    Y - y-values (numpy array, same length as X)
    Ypred - predicted values (numpy array, same length as X)
    slope - slope of the regression line (used for plot title), default: 1
    intercept - intercept of the regression line (used for plot title), default: 0
    """
    plt.clf()
    for i in range(len(X)):
        plt.plot([X[i],X[i]],[Y[i],Ypred[i]],'k:',lw='.8')
    plt.plot(X,Ypred,'.-')
    plt.plot(X,Y,".")
    text='$f(x)={:.3f}x+{:.3f}$'.format(slope,intercept)
    plt.title(text)
    plt.xlabel('x-values')
    plt.ylabel('y-values')


def sumOfSquaredErrors(Y,YPred):
    """ calculates the sum of squared errors
    Y - the values (numpy array)
    YPred - the predicted values (numpy array, same length as Y)
    """
    errors=((Y-YPred)**2).sum()
    return errors