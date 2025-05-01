# 🫀 Heart Disease Prediction

## Decision Tree:

A Decision Tree is a supervised machine learning algorithm used for both classification and regression tasks. It models decisions and their possible consequences in the form of a tree-like structure.

🌳 How It Works:
Root Node: Starts with the entire dataset and selects the best feature to split on based on some metric (like Gini Impurity or Information Gain).

Internal Nodes: Represent decisions based on a feature.

Leaf Nodes: Represent the final outcome (class label or value).

At each node, the algorithm picks the feature and threshold that best splits the data into "pure" subsets (i.e., subsets where most instances belong to a single class).

📌 Example:
Imagine a medical diagnosis based on:

Age

Blood pressure

Cholesterol 

       [Age > 50?]
         /    \
      Yes     No
     /          \
[Chol > 200?]   [Healthy]
   /     \
 Yes     No
[Heart] [Healthy]

✅ Pros:
Easy to understand and interpret.

Requires little data preprocessing (no scaling or normalization needed).

Can handle both numerical and categorical data.

❌ Cons:
Prone to overfitting, especially with deep trees.

Small changes in data can lead to different trees (not stable).


## Random Forest

A Random Forest is an ensemble learning algorithm that builds multiple decision trees and combines their outputs to make more accurate and stable predictions. It's commonly used for both classification and regression tasks.

🌲 How It Works:
Bootstrap Aggregation (Bagging):

It creates many subsets of the training data by randomly sampling with replacement.

Each subset is used to train a separate decision tree.

Feature Randomness:

At each split in a tree, only a random subset of features is considered.

This decorrelates the trees and improves generalization.

Voting/Averaging:

For classification: Each tree votes; the majority class wins.

For regression: The predictions are averaged.

🔍 Example:
If you use a random forest with 100 trees to classify whether someone has heart disease, and:

70 trees predict "Yes"

30 trees predict "No"

Then the final prediction is "Yes".

✅ Pros:
High accuracy.

Resistant to overfitting.

Works well with unscaled and missing data.

Handles both numerical and categorical data.

❌ Cons:
Less interpretable than a single decision tree.

Can be slower to predict (especially with many trees).

Larger memory usage.


## 🔹 1. Load and Inspect the Data

```python
import pandas as pd

df = pd.read_csv(r"C:\Users\USER\Desktop\AI & ML INTERNSHIP\day5\heart.csv")
print(df.head())
print(df.info())
print(df.describe())
```

## 🔹 2. Preprocessing
*Encode categorical features if needed (e.g., cp, thal, slope)

*Normalize numeric features (optional for trees, but useful for comparisons)

```python
from sklearn.preprocessing import StandardScaler

X = df.drop('target', axis=1)
y = df['target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

## 🔹 3. Split Dataset

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
```

## 🌳 4. Decision Tree with Grid Search

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

param_grid = {'max_depth': [3, 5, 7, None], 'min_samples_split': [2, 5, 10]}
grid_search_dt = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5)
grid_search_dt.fit(X_train, y_train)
```

```python
best_dt = grid_search_dt.best_estimator_
print("Best Decision Tree Params:", grid_search_dt.best_params_)
```

## 📈 5. Evaluate Decision Tree

```python
from sklearn.metrics import classification_report, confusion_matrix

y_pred_dt = best_dt.predict(X_test)
print(confusion_matrix(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt))
```

## 📊 6. Visualize the Tree

```python
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 10))
plot_tree(best_dt, feature_names=X.columns, class_names=["No", "Yes"], filled=True)
plt.show()
```

## 🌲 7. Train and Evaluate Random Forest

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
rf.fit(X_train, y_train)
```

```python
y_pred_rf = rf.predict(X_test)
print("Random Forest Accuracy:", rf.score(X_test, y_test))
print(classification_report(y_test, y_pred_rf))
```

## 🔍 8. Feature Importances

```python
import numpy as np

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10,6))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices])
plt.xticks(range(X.shape[1]), [X.columns[i] for i in indices], rotation=90)
plt.show()
```

## 🔁 9. Cross-Validation Comparison

```python
from sklearn.model_selection import cross_val_score

dt_cv = cross_val_score(best_dt, X, y, cv=5).mean()
rf_cv = cross_val_score(rf, X, y, cv=5).mean()
print("CV Accuracy - Decision Tree:", dt_cv)
print("CV Accuracy - Random Forest:", rf_cv)
```

```python
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 4))
sns.barplot(x=models, y=scores, color="skyblue")  # fixed: direct color
plt.title("Model Cross-Validation Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0.7, 1)
plt.tight_layout()
plt.show()
```

```python
from sklearn.metrics import confusion_matrix

cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues')
plt.title('Random Forest Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
```

```python
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12,10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()
```

```python

```