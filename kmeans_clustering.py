import pandas as pd
from sklearn.cluster import KMeans


# Load the data into a dataframe using pandas
URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-BD0231EN-SkillsNetwork/datasets/customers.csv"
df = pd.read_csv(URL)

# Apply k-means clustering
number_of_clusters = 3
cluster = KMeans(n_clusters=number_of_clusters)
result = cluster.fit_transform(df)

# Print cluster centers
cluster.cluster_centers_

# Make a predictions
df["cluster_number"] = cluster.predict(df)

# Print the cluster numbers and the number of customers in each cluster
df.cluster_number.value_counts()
