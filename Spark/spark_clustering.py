# FindSpark simplifies the process of using Apache Spark with Python
import findspark
findspark.init()

# Import functions/Classes for SparkML
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession

# Suppress warnings generated by your code:
import warnings


def warn(*args, **kwargs):
    pass


warnings.warn = warn
warnings.filterwarnings('ignore')

# Create SparkSession
spark = SparkSession.builder.appName("Clustering using SparkML").getOrCreate()

# Load customers dataset
customer_data = spark.read.csv("customers.csv", header=True, inferSchema=True)

# Each row in this dataset is about a customer. The columns indicate the orders placed
# by a customer for Fresh_food, Milk, Grocery and Frozen_Food
customer_data.printSchema()
customer_data.show(n=5, truncate=False)

# Assemble the features into a single vector column
feature_cols = ['Fresh_Food', 'Milk', 'Grocery', 'Frozen_Food']
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
customer_transformed_data = assembler.transform(customer_data)

# You must tell the KMeans algorithm how many clusters to create out of your data
number_of_clusters = 3

# Create a clustering model
kmeans = KMeans(k = number_of_clusters)

# Train the model
model = kmeans.fit(customer_transformed_data)

# Make predictions on the dataset
predictions = model.transform(customer_transformed_data)

# Display the results
predictions.show(5)

# Display how many customers are there in each cluster
predictions.groupBy('prediction').count().show()

# Stop  the Spark session
spark.stop()