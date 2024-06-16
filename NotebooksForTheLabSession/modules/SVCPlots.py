import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns

from sklearn.svm import SVC

from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plotSimpleScatter(df,name,folder="plots/"):
    """
    The function creates a simple scatter plot and stores it on the drive (.png format).
    Parameters
    df -- pandas data frame
          df[:,0] - x-values
          df[:,1] - y-values
          df[:,2] - class label
    name - name of the file
    folder - folder 
    """
    # plot the data
    col=df.columns
    fig=plt.figure()   
    
    sns.scatterplot(data=df, x=col[0], y=col[1],hue=col[2])
    plt.title("The scatter Plot: "+name)
    fig.savefig(f'{folder}scatterplot_{name}.png',dpi=600)
    
    
def plotScatterWithLinearHyperplane(df,clf,name,folder="plots/"):
    """
    The function creates a simple scatter plot of the data and plots the 
    hyperplane, the margins and the support vectors. 
    The implementation of the function was highly inspired by the example on `sklearn`: 
    https://scikit-learn.org/stable/auto_examples/svm/plot_svm_margin.html
    The plot stored on the drive (.png format).
    Parameters
    df -- pandas data frame
          df[:,0] - x-values
          df[:,1] - y-values
          df[:,2] - class label
    clf -- classifier (trained SVC)
    name - name of the file
    folder - folder 
    """
    cols=df.columns
    
    # get the coefficients: w0 and w1
    w0,w1=clf.coef_[0]
    # and the intercept
    intc=clf.intercept_
    # calculate a and b
    a=-w0/w1
    b=-intc/w1
    # we plot the equation of the hyperplane in the graph
    fText='$f(x)={:.4f}\cdot x+{:.4f}$'.format(float(a),float(b[0]))
    # minimum and maximum values
    xmin=df[cols[0]].min()-1
    xmax=df[cols[0]].max()+1
    ymin=df[cols[1]].min()-2
    ymax=df[cols[1]].max()+2
    # calculate the hyperplane
    xValues=np.linspace(xmin,xmax,20)
    yValues=xValues*a+b
    # calculate the margin
    margin=1/np.sqrt(np.sum(clf.coef_**2))
    yy_down=yValues-np.sqrt(1+a**2)*margin
    yy_up=yValues+np.sqrt(1+a**2)*margin
    # plot the data
    fig=plt.figure()   
    sns.scatterplot(data=df, x=cols[0], y=cols[1],hue=cols[2],s=8)
    plt.plot(xValues,yValues,'gray',label='hyperplane')
    plt.plot(xValues,yy_down,c='gray',linestyle=':', label='hyperplane')
    plt.plot(xValues,yy_up,c='gray',linestyle=':',label='hyperplane')
    plt.scatter(clf.support_vectors_[:,0],clf.support_vectors_[:,1],edgecolors='k',facecolor='none',s=20)
    plt.ylim((ymin,ymax))
    dx=abs((xmax-xmin)/90)
    dy=abs((ymax-ymin)/80)
    plt.text(xmin+dx,ymin+dy, fText,horizontalalignment='left', fontsize=10)
    plt.title("The scatter plot wt SVC: " + name)
    fig.savefig(f'{folder}scatterplotHyper_{name}.png', dpi=600)
