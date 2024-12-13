import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset (Human Age Prediction Synthetic Dataset)
url = "Train.csv"  # Update with the correct path or URL
# Columns based on dataset provided in the error traceback
columns = [
    "Gender", "Height (cm)", "Weight (kg)", "Blood Pressure (s/d)", "Cholesterol Level (mg/dL)", "BMI", "Blood Glucose Level (mg/dL)",
    "Bone Density (g/cm²)", "Vision Sharpness", "Hearing Ability (dB)", "Physical Activity Level", "Smoking Status", "Alcohol Consumption",
    "Diet", "Chronic Diseases", "Medication Use", "Family History", "Cognitive Function", "Mental Health Status", "Sleep Patterns",
    "Stress Levels", "Pollution Exposure", "Sun Exposure", "Education Level", "Income Level", "Age (years)"
]
dataset = pd.read_csv(url, header=0, names=columns)

# Explore dataset
print("First 10 rows of the dataset:")
print(dataset.head(10))

# Handle categorical data for Gender (if needed)
dataset['Gender'] = dataset['Gender'].map({'Male': 0, 'Female': 1})

if "Blood Pressure (s/d)" in dataset.columns:
    bp_split = dataset["Blood Pressure (s/d)"].str.split('/', expand=True)
    dataset["Systolic_BP"] = pd.to_numeric(bp_split[0], errors='coerce')
    dataset["Diastolic_BP"] = pd.to_numeric(bp_split[1], errors='coerce')
    dataset.drop(columns=["Blood Pressure (s/d)"], inplace=True)

dataset = pd.get_dummies(dataset, drop_first=True)
# Ensure target column exists
target_column = "Age (years)"
if target_column not in dataset.columns:
    raise ValueError(f"The target column '{target_column}' is not found in the dataset.")

# Handle missing values
dataset.fillna(dataset.mean(numeric_only=True), inplace=True)
# Split features and labels
X = dataset.drop(columns=[target_column]).values
y = dataset[target_column].values


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# --- Linear Regression Model ---
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Make predictions using Linear Regression
lr_pred = lr_model.predict(X_test)

# Evaluate Linear Regression Model
lr_mse = mean_squared_error(y_test, lr_pred)
lr_r2 = r2_score(y_test, lr_pred)

print("\n Random Forest Regressor Results:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R^2 Score: {r2:.2f}")

print("\nLinear Regression Results:")
print(f"Mean Squared Error: {lr_mse:.2f}")
print(f"R² Score: {lr_r2:.2f}")

#fo i, prediction in enumerate(y_pred[:10], start=1):  # Display first 10 predictions
   #print(f"Prediction {i}: {prediction:.2f}, Actual: {y_test[i-1]}")

# Scale the full feature set
X_scaled = scaler.transform(X)

# Predict ages for the entire dataset
dataset['Predicted Age'] = model.predict(X_scaled)

# Display the dataset with actual and predicted ages
print(dataset[['Age (years)', 'Predicted Age']].head(10))
