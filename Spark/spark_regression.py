# Regression using SparkML : use SparkML to predict the mileage of a car

import findspark

findspark.init()

from pyspark.sql import SparkSession

# import functions/Classes for sparkml
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression

# import functions/Classes for metrics
from pyspark.ml.evaluation import RegressionEvaluator

# Create SparkSession
spark = SparkSession.builder.appName("Regressing using SparkML").getOrCreate()

# Load mpg dataset
mpg_data = spark.read.csv("../Data/mpg.csv", header=True, inferSchema=True)

# Explore the dataset
mpg_data.printSchema()
mpg_data.show(5)

#  Group the input Cols as single column named "features" using VectorAssembler
assembler = VectorAssembler(
    inputCols=[
        "Cylinders",
        "Engine Disp",
        "Horsepower",
        "Weight",
        "Accelerate",
        "Year",
    ],
    outputCol="features",
)
mpg_transformed_data = assembler.transform(mpg_data)

# Display the assembled "features" and the label column "MPG"
mpg_transformed_data.select("features", "MPG").show()

# Split data into training and testing sets : 70% training data, 30% testing data
# The random_state variable "seed" controls the shuffling applied to the data before applying the split.
# Pass the same integer for reproducible output across multiple function calls
(training_data, testing_data) = mpg_transformed_data.randomSplit([0.7, 0.3], seed=42)

# Train a linear regression model
lr = LinearRegression(featuresCol="features", labelCol="MPG")
model = lr.fit(training_data)

# Evaluate the model
# Make predictions on testing data
predictions = model.transform(testing_data)

# R-squared (R2) - higher the value better the performance.
evaluator = RegressionEvaluator(
    labelCol="MPG", predictionCol="prediction", metricName="r2"
)
r2 = evaluator.evaluate(predictions)
print("R Squared =", r2)

# Root Mean Squared Error (RMSE) - lower values indicate better performance.
evaluator = RegressionEvaluator(
    labelCol="MPG", predictionCol="prediction", metricName="rmse"
)
rmse = evaluator.evaluate(predictions)
print("RMSE =", rmse)

# Mean Absolute Error (MAE)-lower values indicate better performance.
evaluator = RegressionEvaluator(
    labelCol="MPG", predictionCol="prediction", metricName="mae"
)
mae = evaluator.evaluate(predictions)
print("MAE =", mae)

# Stop the Spark session
spark.stop()
