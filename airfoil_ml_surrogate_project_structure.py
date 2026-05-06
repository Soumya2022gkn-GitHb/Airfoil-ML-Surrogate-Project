import os


# ========================================
# PROJECT NAME
# ========================================
project_name = "airfoil_ml_surrogate"


# ========================================
# FOLDERS TO CREATE
# ========================================
folders = [
    "data_generation",
    "dataset",
    "training",
    "models",
    "app",
    "graphs"
]


# ========================================
# FILES TO CREATE
# ========================================
files = [
    "data_generation/generate_airfoils.py",
    "data_generation/run_xfoil.py",

    "dataset/airfoil_data.csv",

    "training/train_model.py",
    "training/evaluate_model.py",

    "models/surrogate_model.pkl",

    "app/app.py",

    "graphs/cl_prediction.png",
    "graphs/cd_prediction.png",
    "graphs/error_distribution.png",

    "README.md"
]


# ========================================
# CREATE ROOT PROJECT DIRECTORY
# ========================================
os.makedirs(
    project_name,
    exist_ok=True
)

print(f"\nProject folder created: {project_name}")


# ========================================
# CREATE SUBFOLDERS
# ========================================
for folder in folders:

    folder_path = os.path.join(
        project_name,
        folder
    )

    os.makedirs(
        folder_path,
        exist_ok=True
    )

    print(f"Folder created: {folder_path}")


# ========================================
# CREATE FILES
# ========================================
for file in files:

    file_path = os.path.join(
        project_name,
        file
    )

    # Create empty file
    with open(file_path, "w") as f:
        pass

    print(f"File created: {file_path}")


# ========================================
# SUCCESS MESSAGE
# ========================================
print("\nAirfoil ML Surrogate project structure created successfully!")