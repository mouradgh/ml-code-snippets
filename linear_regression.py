# This code will create a linear regression that predict the mileage of a car based on horsepower and weight

import pandas as pd
from sklearn.linear_model import LinearRegression

# Load the data into a dataframe using pandas
URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-BD0231EN-SkillsNetwork/datasets/mpg.csv"
df = pd.read_csv(URL)

# Look at some sample rows and find out the number of rows and columns in the dataset
df.sample(5)
df.shape

# Create a scatter plot of Horsepower versus mileage to visualize the relationship between them
df.plot.scatter(x = "Horsepower", y = "MPG")

# Target is the value that our machine learning model needs to predict
target = df["MPG"]

# Features are the values our machine learning model learns from
features = df[["Horsepower","Weight"]]

# Create a linear regression model
lr = LinearRegression()

# Train/fit the model
lr.fit(features,target)

# Evaluate the model. Higher the number, better the score
lr.score(features,target)

# Predict the mileage for a car with HorsePower = 100 and Weight = 2000
lr.predict([[100,2000]])