# FindSpark simplifies the process of using Apache Spark with Python
import findspark

findspark.init()

# Import SparkSession & SparkContext
from pyspark.sql import SparkSession

# Create SparkSession
spark = SparkSession.builder.appName("Hello World").getOrCreate()

# Load mpg dataset
mpg_data = spark.read.csv("../Data/mpg.csv", header=True, inferSchema=True)

# Explore the data set
mpg_data.printSchema()

# Stop the Spark session
spark.stop()
