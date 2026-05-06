import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    mean_squared_error,
    r2_score
)

# ========================================
# PROJECT PATHS
# ========================================
BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATASET_PATH = os.path.join(
    BASE_DIR,
    "../dataset/airfoil_data.csv"
)

MODELS_DIR = os.path.join(
    BASE_DIR,
    "../models"
)

# Create models directory
os.makedirs(
    MODELS_DIR,
    exist_ok=True
)

# ========================================
# LOAD DATASET
# ========================================
df = pd.read_csv(DATASET_PATH)

print("Dataset loaded successfully!")
print(df.head())

# ========================================
# ENCODE AIRFOIL TYPES
# ========================================
encoder = LabelEncoder()

df["airfoil_encoded"] = encoder.fit_transform(
    df["airfoil"]
)

# ========================================
# FEATURES & TARGETS
# ========================================
X = df[[
    "airfoil_encoded",
    "angle_of_attack",
    "reynolds_number",
    "velocity"
]]

Y_cl = df["Cl"]
Y_cd = df["Cd"]

# ========================================
# TRAIN / TEST SPLIT
# ========================================
X_train, X_test, y_train_cl, y_test_cl = train_test_split(
    X,
    Y_cl,
    test_size=0.2,
    random_state=42
)

_, _, y_train_cd, y_test_cd = train_test_split(
    X,
    Y_cd,
    test_size=0.2,
    random_state=42
)

# ========================================
# RANDOM FOREST MODELS
# ========================================
cl_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

cd_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

# ========================================
# TRAIN MODELS
# ========================================
print("\nTraining Lift Coefficient Model...")
cl_model.fit(X_train, y_train_cl)

print("Training Drag Coefficient Model...")
cd_model.fit(X_train, y_train_cd)

print("\nModel training complete!")

# ========================================
# PREDICTIONS
# ========================================
y_pred_cl = cl_model.predict(X_test)

y_pred_cd = cd_model.predict(X_test)

# ========================================
# EVALUATION METRICS
# ========================================
cl_rmse = mean_squared_error(
    y_test_cl,
    y_pred_cl
) ** 0.5

cd_rmse = mean_squared_error(
    y_test_cd,
    y_pred_cd
) ** 0.5

cl_r2 = r2_score(
    y_test_cl,
    y_pred_cl
)

cd_r2 = r2_score(
    y_test_cd,
    y_pred_cd
)

print("\n========== MODEL PERFORMANCE ==========")

print(f"Lift Model RMSE : {cl_rmse:.4f}")
print(f"Lift Model R²   : {cl_r2:.4f}")

print(f"Drag Model RMSE : {cd_rmse:.5f}")
print(f"Drag Model R²   : {cd_r2:.4f}")

# ========================================
# SAVE MODELS
# ========================================
cl_model_path = os.path.join(
    MODELS_DIR,
    "cl_model.pkl"
)

cd_model_path = os.path.join(
    MODELS_DIR,
    "cd_model.pkl"
)

encoder_path = os.path.join(
    MODELS_DIR,
    "label_encoder.pkl"
)

joblib.dump(
    cl_model,
    cl_model_path
)

joblib.dump(
    cd_model,
    cd_model_path
)

joblib.dump(
    encoder,
    encoder_path
)

# ========================================
# SUCCESS MESSAGE
# ========================================
print("\nModels saved successfully!")

print("\nSaved files:")
print(cl_model_path)
print(cd_model_path)
print(encoder_path)