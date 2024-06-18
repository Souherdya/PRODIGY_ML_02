import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os
print(os.getcwd())

dataSet = pd.read_csv("Mall_Customers.csv")
customer_purchase_DataSet = dataSet.groupby(["CustomerID","Annual Income (k$)"])["Spending Score (1-100)"].sum().unstack().fillna(0)


#data grouped

scaler = StandardScaler()
scaled_data = scaler.fit_transform(customer_purchase_DataSet)

#data scaled

num_clusters = 5 # Number of clusters (tune this as necessary)
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
customer_clusters = kmeans.fit_predict(scaled_data)

customer_purchase_DataSet['cluster'] = customer_clusters
print(customer_purchase_DataSet)

plt.figure(figsize=(10, 7))
plt.scatter(scaled_data[:, 0], scaled_data[:, 1], c=customer_clusters, cmap='viridis')
plt.xlabel('Scaled Feature 1')
plt.ylabel('Scaled Feature 2')
plt.title('Customer Clusters')
plt.colorbar(label='Cluster')
plt.show()

