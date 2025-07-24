# 1. Import Libraries
import os
import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score

# 2. Download Dataset using KaggleHub
dataset_path = kagglehub.dataset_download("cherngs/heart-disease-cleveland-uci")

# 3. Use the correct CSV file
csv_file = os.path.join(dataset_path, "heart_cleveland_upload.csv")

# 4. Load the CSV into a DataFrame
df = pd.read_csv(csv_file)

# 5. Basic Info
print(" First 5 rows:\n", df.head())
print("\n Dataset Info:")
print(df.info())
print("\n Summary Stats:\n", df.describe())
print("\n Missing values:\n", df.isnull().sum())

# 6. Visualization: Countplot of condition
plt.figure(figsize=(6, 4))
sns.countplot(x='condition', data=df)
plt.title("Heart Disease Presence (1 = Yes, 0 = No)")
plt.xlabel("Heart Disease")
plt.ylabel("Count")
plt.show()

# 7. Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# 8. Prepare Features and Labels
X = df.drop('condition', axis=1)
y = df['condition']

# 9. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 10. Model Training
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 11. Model Prediction
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"\n Accuracy: {accuracy:.2f}")

# 12. Confusion Matrix
cm = confusion_matrix(y_test, predictions)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 13. ROC Curve
y_probs = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_probs)
auc_score = roc_auc_score(y_test, y_probs)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.show()

# 14. Feature Importance
importance = pd.Series(model.coef_[0], index=X.columns)
importance.sort_values(ascending=False).plot(kind='bar', figsize=(10, 6), color='orange')
plt.title(" Feature Importance (Logistic Regression)")
plt.ylabel("Model Coefficient")
plt.grid(True)
plt.show()
