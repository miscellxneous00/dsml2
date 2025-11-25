# Write a program to do the following: You have given a collection of 8
# points. P1=[0.1,0.6] P2=[0.15,0.71] P3=[0.08,0.9] P4=[0.16, 0.85]
# P5=[0.2,0.3] P6=[0.25,0.5] P7=[0.24,0.1] P8=[0.3,0.2]. Perform the k-mean
# clustering with initial centroids as m1=P1 =Cluster#1=C1 and
# m2=P8=cluster#2=C2. Answer the following 1] Which cluster does P6
# belong to? 2] What is the population of a cluster around m2? 3] What is
# the updated value of m1 and m2?

# Write a program to do the following: You have given a collection of 8
# points. P1=[2, 10] P2=[2, 5] P3=[8, 4] P4=[5, 8] P5=[7,5] P6=[6, 4] P7=[1, 2]
# P8=[4, 9]. Perform the k-mean clustering with initial centroids as m1=P1
# =Cluster#1=C1 and m2=P4=cluster#2=C2, m3=P7 =Cluster#3=C3. Answer
# the following 1] Which cluster does P6 belong to? 2] What is the
# population of a cluster around m3? 3] What is the updated value of m1,
# m2, m3?

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

data=[[0.1,0.6],[0.15,0.71],[0.08,0.9],[0.16, 0.85],[0.2,0.3],[0.25,0.5],[0.24,0.1],[0.3,0.2]]

df=pd.DataFrame(data,columns=['x','y'])

df

plt.scatter(data=df,x='x',y='y')

from sklearn.preprocessing import MinMaxScaler

scalar=MinMaxScaler()

scaled_data=scalar.fit_transform(df)

from sklearn.cluster import KMeans

initial_clusters = np.array([[0.1,0.6],[0.3,0.2]])

model = KMeans(n_clusters=2,init=initial_clusters,n_init=1,verbose=2)

y_pred=model.fit_predict(df)

df['cluster']=y_pred

df

cluster_1 = df[df['cluster']==0]
plt.scatter(data=cluster_1,x='x',y='y',color='green')

cluster_2 = df[df['cluster']==1]
plt.scatter(data=cluster_2,x='x',y='y',color='red')

plt.show()

idx=df[(df['x']==0.25) & (df['y']==0.5)].index
print('p6 belongs to cluster' ,df.loc[idx,'cluster'].item())

idx=df[(df['x']==0.3) & (df['y']==0.2)].index
m2_cluster=df.loc[idx,'cluster'].item()
population_around_m2=df[df['cluster']==m2_cluster].shape[0]

print('population of a cluster around m2(0.3,0.2) is',population_around_m2)

updated_clusters=model.cluster_centers_
print('updated clusters are :')
for i,center in enumerate(updated_clusters):
    print(f"Cluster {i} centered at : {center}")