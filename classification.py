# This code will create a classifier that can classify if a person has diabetes

import pandas as pd
from sklearn.linear_model import LogisticRegression

# import functions for train test split
from sklearn.model_selection import train_test_split

# functions for metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# Load the data into a dataframe using pandas
URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-BD0231EN-SkillsNetwork/datasets/diabetes.csv"
df = pd.read_csv(URL)

# Look at some sample rows and find out the number of rows and columns in the dataset
df.sample(5)
df.shape

# Plot the types and count of species
df.Species.value_counts().plot.bar()

# Target is the value that our machine learning model needs to predict
y = df["Outcome"]

# Features are the values our machine learning model learns from
X = df[
    [
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age",
    ]
]

# Split the data set into 70% training data and 30% testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=40
)

# Create a Logistic Regression model
classifier = LogisticRegression()

# Train/Fit the model
classifier.fit(X_train, y_train)

# Evaluate the model
classifier.score(X_test, y_test)

# predict the species of a flower with SepalLengthCm = 5.4, SepalWidthCm = 2.6, PetalLengthCm = 4.1, PetalWidthCm = 1.3
classifier.predict([[5.4, 2.6, 4.1, 1.3]])

# Metrics for classification
# To compute the detailed metrics we need two values, the original values and the predicted values
original_values = y_test
predicted_values = classifier.predict(X_test)

# Precision - higher the value, better the model
precision_score(original_values, predicted_values)

# Recall - higher the value, better the model
recall_score(original_values, predicted_values)

# F1 score - higher the value, better the model
f1_score(original_values, predicted_values)

# Confusion matrix
confusion_matrix(original_values, predicted_values)
