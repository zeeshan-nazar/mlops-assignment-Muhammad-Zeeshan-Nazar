import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Load dataset
data = pd.read_csv("C:/Users/Zeeshan/Downloads/PhD/1st Semester Fall 2026/MLOps/Assignments/mlops-assignment-Muhammad-Zeeshan-Nazar/data/House_Rent_Dataset.csv")

print("Dataset loaded successfully!")
print(data.head())

# Separate features and target
X = data[["Size", "Bathroom"]]
y = data["Rent"]

# Create and train the model
model = polynomialRegression()
model.fit(X, y)

# Predictions
y_pred = model.predict(X)

# Model parameters
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
print("R² score:", model.score(X, y))

# Save trained model
joblib.dump(model, "model/trained_model.pkl")

print("Model trained successfully!")
print("Model saved to model/trained_model.pkl")
