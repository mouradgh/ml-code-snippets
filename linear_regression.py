# This code will create a linear regression that predict the mileage of a car based on horsepower and weight

import pandas as pd
from sklearn.linear_model import LinearRegression

# import functions for train test split
from sklearn.model_selection import train_test_split

# import functions for metrics
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from math import sqrt

# Load the data into a dataframe using pandas
URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-BD0231EN-SkillsNetwork/datasets/mpg.csv"
df = pd.read_csv(URL)

# Look at some sample rows and find out the number of rows and columns in the dataset
df.sample(5)
df.shape

# Create a scatter plot of Horsepower versus mileage to visualize the relationship between them
df.plot.scatter(x="Horsepower", y="MPG")

# Target is the value that our machine learning model needs to predict
y = df["MPG"]

# Features are the values our machine learning model learns from
X = df[["Horsepower", "Weight"]]

# Split the data set into 70% training data and 30% testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

# Create a linear regression model
lr = LinearRegression()

# Train/fit the model
lr.fit(X_train, y_train)

# Evaluate the model. Higher the number, better the score
lr.score(X_test, y_test)

# Predict the mileage for a car with HorsePower = 100 and Weight = 2000
lr.predict([[100, 2000]])

# Metrics for regression
# To compute the detailed metrics we need two values, the original mileage and the predicted mileage.
original_values = y_test
predicted_values = lr.predict(X_test)

# R squared - higher the value, better the model
r2_score(original_values, predicted_values)

# Mean Squared Error - lower the value, better the model
mean_squared_error(original_values, predicted_values)

# Root Mean Squared Error - lower the value, better the model
sqrt(mean_squared_error(original_values, predicted_values))

# Mean Absolute Error - lower the value, better the model
mean_absolute_error(original_values, predicted_values)
