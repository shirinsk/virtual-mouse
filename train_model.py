import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib  # For saving and loading the model

# Generate sample eye movement data (you can replace this with real collected data)
data = {
    "eye_x": [0.4, 0.5, 0.6, 0.7, 0.8],
    "eye_y": [0.3, 0.4, 0.5, 0.6, 0.7],
    "cursor_x": [100, 200, 300, 400, 500],
    "cursor_y": [150, 250, 350, 450, 550]
}

# Convert data to DataFrame
df = pd.DataFrame(data)

# Save as CSV (optional)
df.to_csv("eye_movement_data.csv", index=False)

# Separate features (eye positions) and labels (cursor positions)
X = df[["eye_x", "eye_y"]]  # Input: Eye movement
y = df[["cursor_x", "cursor_y"]]  # Output: Cursor movement

# Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, "eye_tracking_model.pkl")

print("✅ Model trained and saved successfully!")
