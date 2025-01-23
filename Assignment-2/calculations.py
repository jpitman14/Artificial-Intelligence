import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Loads dataset
data = pd.read_csv("Assignment-2\\Data\\energy_data.csv")

# Preprocessing
label_encoder = LabelEncoder()

# List of variables to analyze
features = ['Square Footage', 'Number of Occupants', 'Appliances Used', 'Average Temperature',]