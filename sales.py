import pandas as pd

# Load dataset
df = pd.read_csv('sales.csv')

# Display first few rows
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Fill missing values if necessary (this depends on your actual dataset)
df.fillna(0, inplace=True)

# Convert date columns to datetime format
df['Order Date'] = pd.to_datetime(df['Order Date'])
df['Ship Date'] = pd.to_datetime(df['Ship Date'])

# Remove duplicates if any
df.drop_duplicates(inplace=True)

# Calculate additional fields (e.g., profit margin)
df['Profit Margin'] = df['Total Profit'] / df['Total Revenue']

# Calculate KPIs
total_revenue = df['Total Revenue'].sum()
total_profit = df['Total Profit'].sum()
avg_profit_margin = df['Profit Margin'].mean()

# Display KPIs
print(f"Total Revenue: {total_revenue}")
print(f"Total Profit: {total_profit}")
print(f"Average Profit Margin: {avg_profit_margin:.2%}")

# Perform groupby operation for OLAP-like analysis (Total Revenue and Profit by Region and Item Type)
olap_analysis = df.groupby(['Region', 'Item Type'])[['Total Revenue', 'Total Profit']].sum().reset_index()

# Display the results of the OLAP-like analysis
print("\nOLAP Analysis (Total Revenue and Profit by Region and Item Type):\n", olap_analysis)

from sklearn.model_selection import train_test_split

# Define feature variables (X) and target variable (y)
X = df[['Units Sold', 'Unit Price', 'Unit Cost']]
y = df['Total Revenue']

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Initialize the model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate Mean Squared Error to evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

import matplotlib.pyplot as plt

# Plot true vs predicted values
plt.scatter(y_test, y_pred)
plt.xlabel("True Total Revenue")
plt.ylabel("Predicted Total Revenue")
plt.title("True vs Predicted Total Revenue")
plt.show()

# Save the final predictions and KPIs for further analysis
predictions_df = pd.DataFrame({'True Revenue': y_test, 'Predicted Revenue': y_pred})
predictions_df.to_csv('sales_predictions.csv', index=False)

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Initialize the Decision Tree model
dt_model = DecisionTreeRegressor(random_state=42)


# Train the model
dt_model.fit(X_train, y_train)

# Make predictions
y_pred_dt = dt_model.predict(X_test)

# Evaluate the model
mae_dt = mean_absolute_error(y_test, y_pred_dt)
mse_dt = mean_squared_error(y_test, y_pred_dt)
r2_dt = r2_score(y_test, y_pred_dt)

print("Decision Tree Regression:")
print(f"MAE: {mae_dt}, MSE: {mse_dt}, R²: {r2_dt}")


from sklearn.ensemble import RandomForestRegressor

# Initialize the Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions
y_pred_rf = rf_model.predict(X_test)

# Evaluate the model
mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print("Random Forest Regression:")
print(f"MAE: {mae_rf}, MSE: {mse_rf}, R²: {r2_rf}")

from sklearn.ensemble import GradientBoostingRegressor

# Initialize the Gradient Boosting model
gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

# Train the model
gb_model.fit(X_train, y_train)

# Make predictions
y_pred_gb = gb_model.predict(X_test)

# Evaluate the model
mae_gb = mean_absolute_error(y_test, y_pred_gb)
mse_gb = mean_squared_error(y_test, y_pred_gb)
r2_gb = r2_score(y_test, y_pred_gb)

print("Gradient Boosting Regression:")
print(f"MAE: {mae_gb}, MSE: {mse_gb}, R²: {r2_gb}")

import xgboost as xgb

# Initialize the XGBoost model
xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

# Train the model
xgb_model.fit(X_train, y_train)

# Make predictions
y_pred_xgb = xgb_model.predict(X_test)

# Evaluate the model
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)

print("XGBoost Regression:")
print(f"MAE: {mae_xgb}, MSE: {mse_xgb}, R²: {r2_xgb}")

from sklearn.svm import SVR

# Initialize the SVR model
svr_model = SVR(kernel='rbf', C=100, gamma=0.1)

# Train the model
svr_model.fit(X_train, y_train)

# Make predictions
y_pred_svr = svr_model.predict(X_test)

# Evaluate the model
mae_svr = mean_absolute_error(y_test, y_pred_svr)
mse_svr = mean_squared_error(y_test, y_pred_svr)
r2_svr = r2_score(y_test, y_pred_svr)

print("Support Vector Regression:")
print(f"MAE: {mae_svr}, MSE: {mse_svr}, R²: {r2_svr}")

rf_model = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42)
# Train the model
rf_model.fit(X_train, y_train)

# Make predictions
y_pred_rf = rf_model.predict(X_test)

# Evaluate the model
mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print("Random Forest Regression:")
print(f"MAE: {mae_rf}, MSE: {mse_rf}, R²: {r2_rf}")


