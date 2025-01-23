import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Loads dataset
data = pd.read_csv("Assignment-2\\Data\\energy_data.csv")

# Encoding categorical variables
label_encoder = LabelEncoder()
data['Day of Week'] = label_encoder.fit_transform(data['Day of Week'])

# Define features and target variable
X = data[['Square Footage', 'Number of Occupants', 'Appliances Used', 'Average Temperature', 'Day of Week']]
y = data['Energy Consumption']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Regressor Model
random_forest_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
random_forest_regressor.fit(X_train, y_train)

# Predictions
y_pred = random_forest_regressor.predict(X_test)

# Model Evaluation
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# Prints calculations
print("Random Forest Regressor Model")
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R-squared: {r2}")
print(f"Mean Absolute Error: {mae}")

# Plotting Actual vs Predicted values
plt.scatter(y_test, y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel("Actual Energy Consumption")
plt.ylabel("Predicted Energy Consumption")
plt.title("Random Forest Regressor: Actual vs Predicted Energy Consumption")
plt.show()