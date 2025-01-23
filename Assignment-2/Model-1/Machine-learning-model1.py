import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("Assignment-2\\Data\\energy_data.csv")

# Encoding categorical variables
label_encoder = LabelEncoder()
data['Day of Week'] = label_encoder.fit_transform(data['Day of Week'])

# Define features and target variable
X = data[['Square Footage', 'Number of Occupants', 'Appliances Used', 'Average Temperature', 'Day of Week']]
y = data['Energy Consumption']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression Model
linear_regressor = LinearRegression()
linear_regressor.fit(X_train, y_train)

# Predictions
y_pred = linear_regressor.predict(X_test)

# Plotting Actual vs Predicted values
plt.scatter(y_test, y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel("Actual Energy Consumption")
plt.ylabel("Predicted Energy Consumption")
plt.title("Linear Regression: Actual vs Predicted Energy Consumption")
plt.show()