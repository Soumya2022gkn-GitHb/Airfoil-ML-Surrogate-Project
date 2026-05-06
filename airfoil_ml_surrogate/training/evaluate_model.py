import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
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

GRAPHS_DIR = os.path.join(
    BASE_DIR,
    "../graphs"
)


# ========================================
# CREATE GRAPHS DIRECTORY
# ========================================
os.makedirs(
    GRAPHS_DIR,
    exist_ok=True
)

print(f"Graphs directory ready:\n{GRAPHS_DIR}")


# ========================================
# LOAD DATASET
# ========================================
df = pd.read_csv(DATASET_PATH)

print("\nDataset loaded successfully!")


# ========================================
# LOAD LABEL ENCODER
# ========================================
encoder = joblib.load(
    os.path.join(
        MODELS_DIR,
        "label_encoder.pkl"
    )
)

# Encode airfoil names
df["airfoil_encoded"] = encoder.transform(
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
# LOAD TRAINED MODELS
# ========================================
cl_model = joblib.load(
    os.path.join(
        MODELS_DIR,
        "cl_model.pkl"
    )
)

cd_model = joblib.load(
    os.path.join(
        MODELS_DIR,
        "cd_model.pkl"
    )
)

print("Models loaded successfully!")


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
# GRAPH 1: Cl Prediction
# ========================================
plt.figure(figsize=(8, 6))

plt.scatter(
    y_test_cl,
    y_pred_cl,
    alpha=0.6
)

plt.xlabel("Actual Cl")
plt.ylabel("Predicted Cl")

plt.title(
    "Lift Coefficient Prediction"
)

plt.grid(True)

# Save graph automatically
cl_graph_path = os.path.join(
    GRAPHS_DIR,
    "cl_prediction.png"
)

plt.savefig(cl_graph_path)

plt.show()

print(f"\nSaved graph:\n{cl_graph_path}")


# ========================================
# GRAPH 2: Cd Prediction
# ========================================
plt.figure(figsize=(8, 6))

plt.scatter(
    y_test_cd,
    y_pred_cd,
    alpha=0.6
)

plt.xlabel("Actual Cd")
plt.ylabel("Predicted Cd")

plt.title(
    "Drag Coefficient Prediction"
)

plt.grid(True)

# Save graph automatically
cd_graph_path = os.path.join(
    GRAPHS_DIR,
    "cd_prediction.png"
)

plt.savefig(cd_graph_path)

plt.show()

print(f"\nSaved graph:\n{cd_graph_path}")


# ========================================
# GRAPH 3: Lift Error Distribution
# ========================================
cl_error = y_test_cl - y_pred_cl

plt.figure(figsize=(8, 6))

plt.hist(
    cl_error,
    bins=40,
    alpha=0.7
)

plt.xlabel("Prediction Error")
plt.ylabel("Count")

plt.title(
    "Lift Prediction Error Distribution"
)

plt.grid(True)

# Save graph automatically
error_graph_path = os.path.join(
    GRAPHS_DIR,
    "error_distribution.png"
)

plt.savefig(error_graph_path)

plt.show()

print(f"\nSaved graph:\n{error_graph_path}")


# ========================================
# GRAPH 4: Actual vs Predicted Line
# ========================================
plt.figure(figsize=(8, 6))

plt.plot(
    y_test_cl.values[:100],
    label="Actual Cl"
)

plt.plot(
    y_pred_cl[:100],
    label="Predicted Cl"
)

plt.xlabel("Sample Index")
plt.ylabel("Lift Coefficient")

plt.title(
    "Actual vs Predicted Lift Coefficient"
)

plt.legend()

plt.grid(True)

# Save graph automatically
comparison_graph_path = os.path.join(
    GRAPHS_DIR,
    "actual_vs_predicted_cl.png"
)

plt.savefig(comparison_graph_path)

plt.show()

print(f"\nSaved graph:\n{comparison_graph_path}")


# ========================================
# SUCCESS MESSAGE
# ========================================
print("\n========================================")
print("ALL GRAPHS SAVED SUCCESSFULLY!")
print("========================================")

print("\nGenerated graph files:")
print("- cl_prediction.png")
print("- cd_prediction.png")
print("- error_distribution.png")
print("- actual_vs_predicted_cl.png")