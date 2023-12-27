import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def kernel(point,xmat,k):
    m,n=np.shape(xmat)
    weight=np.mat(np.eye(m))
    for j in range(m):
        diff=point-x[j]
        weight[j,j]=np.exp(diff*diff.T/(-2.0*k**2))
    return weight

def localweight(point,xmat,ymat,k):
    wei=kernel(point,xmat,k)
    w=(x.T*(wei*x)).I*(x.T*(wei*ymat.T))
    return w

def localweightregression(xmat,ymat,k):
    m,n=np.shape(xmat)
    ypred=np.zeros(m)
    for i in range(m):
        ypred[i]=xmat[i]*localweight(xmat[i],xmat,ymat,k)
    return ypred

def graphplot(x,ypred):
    sortindex=x[:,1].argsort(0)
    xsort=x[sortindex][:,0]
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    ax.scatter(bill,tip,color='green')
    ax.plot(xsort[:,1],ypred[sortindex],color='red',linewidth='5')
    plt.xlabel("Total bill")
    plt.ylabel("tip")
    plt.show()
    
data=pd.read_csv("data10_tips.csv")
bill=np.array(data.total_bill)
mbill=np.mat(bill)
tip=np.array(data.tip)
mtip=np.mat(tip)
m=np.shape(mbill)
one=np.mat(np.ones(m))
x=np.hstack((one.T,mbill.T))
ypred=localweightregression(x,mtip,0.5)
graphplot(x,ypred)
