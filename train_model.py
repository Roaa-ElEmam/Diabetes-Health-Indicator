import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import pickle

# Load the dataset
df = pd.read_csv("diabetes_012_health_indicators_BRFSS2015.csv")

# Prepare features and target
X = df.drop('Diabetes_012', axis=1)
y = df['Diabetes_012']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"Training accuracy: {train_score:.4f}")
print(f"Testing accuracy: {test_score:.4f}")

# Save the model
joblib.dump(model, 'diabetes_model.pkl')
# Save the feature order
with open('feature_order.pkl', 'wb') as f:
    pickle.dump(list(X.columns), f)
print("Model saved as 'diabetes_model.pkl'")
print("Feature order saved as 'feature_order.pkl'") 