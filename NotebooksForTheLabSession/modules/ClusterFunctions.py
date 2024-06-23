### Code written by Christina B. Class for the course Introduction to DS and ML, EAH Jena
### Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.

from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.cluster import KMeans

def makeCluster(filename='clusters',samples=100,features=2,nbCenters=2,dev=2,state=10):
    X,y,centers=make_blobs(n_samples=samples, n_features=features, centers=nbCenters,cluster_std=dev, random_state=state,return_centers=True)
    cols=['feat{:02}'.format(i) for i in range(1,features+1)] 
    dfX=pd.DataFrame(X,columns=cols) 
    df2=pd.DataFrame(y,columns=['cluster']) 
    dfXplusCluster=dfX.join(df2)            
    df3=pd.DataFrame(list(range(nbCenters)),columns=['cluster']) 
    df4=pd.DataFrame(data=centers,columns=cols)
    dfCenters=df3.join(df4) 
    dfX.to_csv('../data/{}_data.csv'.format(filename),index=False)  
    dfXplusCluster.to_csv('../data/{}_data_withClusters.csv'.format(filename),index=False)
    dfCenters.to_csv('../data/{}_clusterCenters.csv'.format(filename),index=False) 
    with open('../data/{}_info.txt'.format(filename),"w") as fp:
        fp.write('Dateinames:\n- {f}_data.csv \n- {f}_data_withClusters.csv\n- {f}_clusterCenters.csv\n'.format(f=filename))
        fp.write('\nCreating Cluster with make_blobs()\n\nParameter\n')
        fp.write('   n_samples={}\n   n_features={}\n   centers={}\n   cluster_std={}\n'.format(samples,features,nbCenters,dev))
        fp.write('   random_state={}\n   return_centers=True\n'.format(state))

    
def createClusterFiles():
    names=['example1','example2','example3','example4','example5']
    samples=[40,40,100,100,100]
    features=[2,2,2,2,2]
    centers=[2,2,2,3,4]
    dev=[1,3,2,2,2]
    
    for i in range(len(names)):
        makeCluster(filename=names[i],samples=samples[i],features=features[i],
                    nbCenters=centers[i],dev=dev[i])
        title='{}: {} Clusters, {} samples, stddev: {}'.format(names[i], 
                                                           centers[i],
                                                           samples[i],
                                                           dev[i])
        plot2DClustersFromFile(filename=names[i],save=True,title=title,legend=False)
        df=pd.read_csv('../data/{}_data.csv'.format(names[i]))
        scatterPlot2D(df,save=True,filename=names[i])
        
        
        
def plotClusters(df,model,legend='auto',draw_center=False,annotate_centers=False,save=False,file='file',path='../plots/'):
    """
    Function to plot the clusters:
    Parameters:
    df - is the data. It is specified in form of a pandas data frame
    model - is the trained model. It is an instance of the KMeans estimator trained with the data
    legend - if legend is set to False, no legend is drawn
    draw_center - if this parameter is set to True, the centers of the learned clusters are drawn
    annotate_centers - if set to True, the cluster means are annotated with cluster number and
                       an arrow
    save - if set to True the plot will be saved
    file - if save is True, the plot will be saved to a file named file.png 
    """
    plt.clf()
    cols=df.columns
    labels=model.labels_
    plt.Figure()
    sns.scatterplot(data=df,x=cols[0],y=cols[1],hue=labels,style=labels,legend=legend,palette="deep")
    if draw_center: 
        centers=model.cluster_centers_
        plt.plot(centers[:,0],centers[:,1],'*m',ms=10)
        if annotate_centers:
            for i in range(len(centers)):
                # the following code is based on sklearn help
                plt.annotate(str(i),xy=(centers[i,0],centers[i,1]),xytext=(centers[i,0]-1,centers[i,1]-1),
                             arrowprops = dict(facecolor ='black',
                                               arrowstyle = "->"), 
                             bbox=dict(boxstyle ="round", fc ="0.8"))
    if save:
        plt.savefig('{path}{file}.png',dpi=600)