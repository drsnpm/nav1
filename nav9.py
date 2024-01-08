import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def kernel(point,xmat,k):
    m,n = np.shape(xmat)
    weight = np.mat(np.eye(m))
    for i in range(m):
        diff = point - x[i]
        weight[i,i] = np.exp(diff*diff.T / (-2*k**2))
    return weight

def localweight(point,xmat,ymat,k):
    weight = kernel(point,xmat,k)
    w = (x.T*(weight*x)).I*(x.T*(weight*ymat.T))
    return w

def localweightregression(xmat,ymat,k):
    m,n = np.shape(xmat)
    zeros = np.zeros(m)
    for i in range(m):
        zeros[i] = xmat[i]*localweight(xmat[i],xmat,ymat,k)
    return zeros

def graphplot(x,y):
    sortindex = x[:,1].argsort(0)
    xsort = x[sortindex][:,0]
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(xsort[:,1],y[sortindex],color="red",linewidth="5")
    ax.scatter(bill,tip,color="green")
    plt.xlabel("Total bill")
    plt.ylabel("tip")
    plt.show()
    
data = pd.read_csv("data10_tips.csv")
bill = np.array(data.total_bill)
mbill = np.mat(bill)
tip = np.array(data.tip)
mtip = np.mat(tip)
m = np.shape(mbill)
ones = np.mat(np.ones(m))
x = np.hstack((ones.T,mbill.T))
y = localweightregression(x,mtip,0.5)
graphplot(x,y)
