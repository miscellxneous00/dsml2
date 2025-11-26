# Write a program to cluster a set of points using K-means for IRIS
# dataset. Consider, K=3, clusters. Consider Euclidean distance as the
# distance measure. Randomly initialize a cluster mean as one of the data
# points. Iterate at least for 10 iterations. After iterations are over, print the
# final cluster means for each of the clusters.

import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv('./Iris.csv')

df

from sklearn.cluster import KMeans

model = KMeans(n_clusters=3,init='random',n_init=1,max_iter=10)

X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]

df['cluster']=model.fit_predict(X)

df

print('final cluster means are :')
for i,center in enumerate(model.cluster_centers_):
    print(f"Centroid {i} has centered at {center}")


# Write a program to cluster a set of points using K-means for IRIS
# dataset. Consider, K=4, clusters. Consider Euclidean distance as the
# distance measure. Randomly initialize a cluster mean as one of the data
# points. Iterate at least for 10 iterations. After iterations are over, print the
# final cluster means for each of the clusters.

model_2 = KMeans(n_clusters=4,init='random',n_init=1,max_iter=10)

df['cluster']=model.fit_predict(X)

print('final cluster means are :')
for i,center in enumerate(model.cluster_centers_):
    print(f"Centroid {i} has centered at {center}")