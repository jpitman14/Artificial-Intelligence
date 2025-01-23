import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Loads dataset
data = pd.read_csv("Assignment-2\\Data\\energy_data.csv")

# Preprocessing
label_encoder = LabelEncoder()

# List of linear variables to analyze
features = ['Square Footage', 'Number of Occupants', 'Appliances Used', 'Average Temperature',]

# Generate scatter plots with a line of best fit
for feature in features:
    # Scatter plot
    plt.figure(figsize=(6, 4))
    plt.scatter(data[feature], data['Energy Consumption'], alpha=0.5, color='blue', label='Data points')
    
    # Line of best fit
    x = data[feature]
    y = data['Energy Consumption']
    slope, intercept = np.polyfit(x, y, 1)  # Linear regression to calculate slope and intercept
    line = slope * x + intercept
    plt.plot(x, line, color='red', label='Line of Best Fit', linestyle='--')
    
    # Plot formatting
    plt.xlabel(feature)
    plt.ylabel("Energy Consumption")
    plt.title(f"Relationship between {feature} and Energy Consumption")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Bar plot for building type
plt.figure(figsize=(10, 6))
sns.barplot(data=data, x='Building Type', y='Energy Consumption')
plt.title('Average Energy Consumption by Building Type', fontsize=16)
plt.xlabel('Building Type', fontsize=12)
plt.ylabel('Average Energy Consumption', fontsize=12)
plt.show()