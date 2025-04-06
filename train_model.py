import numpy as np
# from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import joblib

# Load extracted features and labels
X = np.load("features.npy")  
y = np.load("y_labels.npy")  


min_length = min(len(X), len(y))
X = X[:min_length]
y = y[:min_length]

# Split data into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train the regression model
model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=5)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R² Score: {r2:.4f}")

# Save trained model
joblib.dump(model, "drowsiness_model.pkl")
print("Model training complete! Model saved as drowsiness_model.pkl")

# Plot actual vs predicted values
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Drowsiness Score")
plt.ylabel("Predicted Drowsiness Score")
plt.title("Actual vs Predicted Drowsiness Scores")
plt.show()

# Plot residual errors
residuals = y_test - y_pred
plt.hist(residuals, bins=20, edgecolor='black')
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.title("Distribution of Prediction Errors (Residuals)")
plt.show()

importances = model.feature_importances_
feature_names = ["EAR", "MAR", "HTR"]

plt.bar(feature_names, importances)
plt.xlabel("Features")
plt.ylabel("Importance Score")
plt.title("Feature Importance in Drowsiness Detection Model")
plt.show()