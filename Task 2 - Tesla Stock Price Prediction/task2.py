# Step 1: Import required libraries
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Step 2: Load historical stock data from Yahoo Finance
stock = yf.download("TSLA", start="2020-01-01", end="2024-12-31")

# Step 3: Prepare dataset
# Select useful features
data = stock[['Open', 'High', 'Low', 'Volume', 'Close']].copy()

# Create target column: next day's close price
data['Next_Close'] = data['Close'].shift(-1)

# Drop the last row (it has NaN in Next_Close)
data.dropna(inplace=True)

# Define features (X) and target (y)
X = data[['Open', 'High', 'Low', 'Volume']]
y = data['Next_Close']

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Step 4: Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Predict on test set
predictions = model.predict(X_test)

# Step 6: Evaluate performance
mse = mean_squared_error(y_test, predictions)
print(" Mean Squared Error:", mse)

# Step 7: Plot actual vs predicted closing prices
plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label='Actual Close Price', linewidth=2)
plt.plot(predictions, label='Predicted Close Price', linewidth=2)
plt.title(" Actual vs Predicted Closing Prices (Tesla)")
plt.xlabel("Days")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
