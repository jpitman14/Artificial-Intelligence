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

# --- Model 2: Expert System ---
def improved_expert_system(row):
    # Class 0 Rules (cultivar 1)
    if (row['color_intensity'] <= 3.82 and 
        row['ash'] > 3.00 and 
        row['od280/od315_of_diluted_wines'] > 2.8):
        return 0
    if (row['flavanoids'] > 2.5 and 
        row['hue'] > 1.0):
        return 0
        
    # Class 1 Rules (cultivar 2)
    elif (row['color_intensity'] <= 3.82 and 
        row['alcalinity_of_ash'] <= 17.65):
        return 1
    elif (row['proline'] > 750 and 
        row['alcohol'] > 13.0):
        return 1
        
    # Class 2 Rules (cultivar 3)
    elif (row['color_intensity'] > 3.82 and 
        row['flavanoids'] <= 1.58 and 
        row['alcalinity_of_ash'] > 17.65):
        return 2
    else:
        # Default to most common class in training data
        return y_train.mode()[0]  

# Evaluate Expert System's performance
y_pred_expert = X_test.apply(improved_expert_system, axis=1)
evaluate_model("Expert System", y_test, y_pred_expert)

# --- Comparison table of each systems Predictions ---
comparison = pd.DataFrame({
    'True Class': y_test,
    'Decision Tree': y_pred_dt,
    'Expert System': y_pred_expert
})
print("\nComparison Table of Predictions:")
print(comparison)