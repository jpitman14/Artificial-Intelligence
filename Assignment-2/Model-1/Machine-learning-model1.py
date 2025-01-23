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
