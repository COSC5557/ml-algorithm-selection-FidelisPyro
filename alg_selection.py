
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix

# Path to my csv file
path = "/mnt/c/Users/kylel/Programming/School/PracticalML/wine+quality/winequality-red.csv"

# Read csv file into a pandas dataframe
df = pd.read_csv(path, delimiter = ";")

# Calculate the correlation matrix
corr_matrix = df.corr()

# Select the most important features relative to the target
important_features = corr_matrix['quality'].sort_values(ascending = False)[1:6].index.tolist()
print(f"Important features: {important_features}")

# Trying combinations of features with PolynomialFeatures
poly = PolynomialFeatures(degree = 1, include_bias = False)

# Create a new dataframe without the quality column
X = df[important_features]
y = df['quality']
X_poly = poly.fit_transform(X)

# Split the dataset into a training set and a temporary set into 60/40 split
X_train, X_temp, y_train, y_temp = train_test_split(X_poly, y, test_size=0.4, random_state=42)

# Split the temporary set into validation and test set into 20/20 split of total dataset
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# Define the pipelines
pipelines = {
    'rf': Pipeline([('scaler', StandardScaler()), ('model', RandomForestClassifier())]),
    'svc': Pipeline([('scaler', StandardScaler()), ('model', SVC(C = 0.5))]),
    'nn': Pipeline([('scaler', StandardScaler()), ('model', MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, alpha=0.001))]),
    'knn': Pipeline([('scaler', StandardScaler()), ('model', KNeighborsClassifier())]),
}

# Fit the pipelines and evaluate on val set
scores = {}
for model_name, pipeline in pipelines.items():
    pipeline.fit(X_train, y_train)
    score = pipeline.score(X_val, y_val)
    scores[model_name] = score  
    print(f"{model_name}: {score}")
    
    # Predict the validation set results
    y_pred = pipeline.predict(X_val)
    
    # Compute confusion matrix
    confusion = confusion_matrix(y_val, y_pred)
    
    sns.heatmap(confusion, annot=True, cmap="Blues", fmt="d")
    plt.xlabel("Actual Quality")
    plt.ylabel("Predicted Quality")
    plt.title("Confusion Matrix " + model_name)
    plt.savefig('Confusion_matrix_'+ model_name + '.png')
    plt.close()  

# Select best model
best_name, best_score = max(scores.items(), key=lambda x: x[1])
best_model = pipelines[best_name]

# Predict the test set results
y_pred_best = best_model.predict(X_test)

best_confusion = confusion_matrix(y_test, y_pred_best)

# Best model evaluation on test set
test_score = best_model.score(X_test, y_test)
print(f"Best model: {best_name} with test score: {test_score}")
sns.heatmap(best_confusion, annot=True, cmap="BuPu", fmt="d")
plt.xlabel("Actual Quality")
plt.ylabel("Predicted Quality")
plt.title("Confusion Matrix Best")
plt.savefig('Confusion_matrix_Best.png')
plt.close()  