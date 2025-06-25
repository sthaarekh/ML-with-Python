import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

customer_data = pd.read_csv('/Users/sthaarekh/Documents/ /                     /Python/Project6/Mall_Customers.csv')
print(customer_data.isnull().sum())

#choosing the annual income and spending score
x = customer_data.iloc[:,[3,4]].values
print(x)

#choosing the number of clusters
# WCSS --> within cluster sum of squares

#finding wcss value for different number of clusters
wcss = []
for i in range (1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

#plot an elbow graph
sns.set()
plt.plot(range(1,11), wcss)
plt.title("The elbow point graph")
plt.xlabel("No of clusters")
plt.ylabel("WCSS")
plt.show()

#from the graph the optimum no of cluster is 5

# Training the KMeans Clustering Model

kmeans = KMeans(n_clusters=5, init='k-means++', random_state=0)
y = kmeans.fit_predict(x)
print(y)

#visualiing the clusters
plt.figure(figsize=(10,10))
plt.scatter(x[y==0,0],x[y==0,1], s=50, c='green', label='Cluster 1')
plt.scatter(x[y==1,0],x[y==1,1], s=50, c='red', label='Cluster 2')
plt.scatter(x[y==2,0],x[y==2,1], s=50, c='blue', label='Cluster 3')
plt.scatter(x[y==3,0],x[y==3,1], s=50, c='yellow', label='Cluster 4')
plt.scatter(x[y==4,0],x[y==4,1], s=50, c='violet', label='Cluster 5')

# plot the centroids
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=100, c='cyan', label='Centroids')

plt.title('Customer Groups')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.show()
