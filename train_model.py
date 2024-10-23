import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load your sales dataset
df = pd.read_csv('sales.csv')  # Update this path as necessary

# Preprocess the data (example)
# Convert Order Date to datetime if not already done
df['Order Date'] = pd.to_datetime(df['Order Date'])

# Extract features and target variable
X = df[['Units Sold', 'Unit Price', 'Unit Cost']]  # Example features
y = df['Total Revenue']  # Target variable

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)

# Save the model
with open('random_forest_model.pkl', 'wb') as file:
    pickle.dump(rf_model, file)

print("Model has been trained and saved as random_forest_model.pkl")
