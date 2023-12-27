import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import datasets
import pandas as pd
import numpy as np
iris=datasets.load_iris()
x=pd.DataFrame(iris.data)
x.columns=['sepal_length','sepal_width','petal_length','petal_width']
y=pd.DataFrame(iris.target)
y.columns=["Targets"]
model=KMeans(n_clusters=3)
model.fit(x)
plt.figure(figsize=(14,14))
colormap=np.array(['red','lime','black'])
plt.subplot(2,2,1)
plt.scatter(x.petal_length,x.petal_width,c=colormap[y.Targets],s=40)
plt.title("real custer")
plt.xlabel("petal_length")
plt.ylabel("petal_width")
plt.subplot(2,2,2)
plt.scatter(x.petal_length,x.petal_width,c=colormap[model.labels_],s=40)
plt.title("kmeans custer")
plt.xlabel("petal_length")
plt.ylabel("petal_width")

from sklearn import preprocessing
scaler=preprocessing.StandardScaler()
scaler.fit(x)
xsa=scaler.transform(x)
xs=pd.DataFrame(xsa,columns=x.columns)
from sklearn.mixture import GaussianMixture
gmm=GaussianMixture(n_components=3)
gmm.fit(xs)
gmm_y=gmm.predict(xs)
plt.subplot(2,2,3)
plt.scatter(x.petal_length,x.petal_width,c=colormap[gmm_y],s=40)
plt.title("gmm custer")
plt.xlabel("petal_length")
plt.ylabel("petal_width")
plt.show()