import pandas as pd
from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix)
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

# Load dataset
data = load_wine()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')

# Split data into a training and test set (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

def evaluate_model(name, y_true, y_pred):
    """function to calculate and display metrics"""
    print(f"\n{name} Performance:")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Macro-average Precision:", precision_score(y_true, y_pred, average='macro'))
    print("Macro-average Recall:", recall_score(y_true, y_pred, average='macro'))
    print("Macro-average F1:", f1_score(y_true, y_pred, average='macro'))
    
    # Confusion matrix visualization
    # calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred) 
    plt.figure(figsize=(6,4))
    # creates the heatmap with the blue colour pallette
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=data.target_names,
                yticklabels=data.target_names)
    plt.title(f'{name} Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

# --- Model 1: Decision Tree ---
dt_model = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_model.fit(X_train, y_train)

y_pred_dt_proba = dt_model.predict_proba(X_test)
y_pred_dt = dt_model.predict(X_test)

# Extract and print rules that the tree learned
tree_rules = export_text(dt_model, feature_names=list(X.columns))
print("Decision Tree Rules:\n", tree_rules)

# Evaluate Decision Tree
y_pred_dt = dt_model.predict(X_test)
evaluate_model("Decision Tree", y_test, y_pred_dt)