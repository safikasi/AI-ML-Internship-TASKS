# Step 1: Import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Step 2: Load the Iris dataset
df = sns.load_dataset("iris")

# Step 3: Print shape, columns, and first few rows
print("Shape of dataset:", df.shape)
print("\nColumn names:", df.columns)
print("\nFirst 5 rows:\n", df.head())

# Step 4: Summary Information
print("\nData Info:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe())

# Step 5: Scatter Plot (to see relationship between sepal_length and sepal_width)
sns.scatterplot(data=df, x="sepal_length", y="sepal_width", hue="species")
plt.title("Sepal Length vs Sepal Width")
plt.show()

# Step 6: Histograms (distribution of each numeric feature)
df.hist(figsize=(10, 8), edgecolor='black')
plt.suptitle("Histogram of All Features")
plt.tight_layout()
plt.show()

# Step 7: Box Plots (to detect outliers)
plt.figure(figsize=(10, 6))
sns.boxplot(data=df)
plt.title("Box Plot for Feature Outlier Detection")
plt.show()
