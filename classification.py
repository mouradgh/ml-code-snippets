# This code will create a classifier that can classify the various species of flowers

import pandas as pd
from sklearn.linear_model import LogisticRegression

# Load the data into a dataframe using pandas
URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-BD0231EN-SkillsNetwork/datasets/iris.csv"
df = pd.read_csv(URL)

# Look at some sample rows and find out the number of rows and columns in the dataset
df.sample(5)
df.shape

# Plot the types and count of species
df.Species.value_counts().plot.bar()

# Target is the value that our machine learning model needs to predict
target = df["Species"]

# Features are the values our machine learning model learns from
features = df[["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]]

# Create a Logistic Regression model
classifier = LogisticRegression()

# Train/Fit the model
classifier.fit(features,target)

# Evaluate the model
classifier.score(features,target)

# predict the species of a flower with SepalLengthCm = 5.4, SepalWidthCm = 2.6, PetalLengthCm = 4.1, PetalWidthCm = 1.3
classifier.predict([[5.4,2.6,4.1,1.3]])