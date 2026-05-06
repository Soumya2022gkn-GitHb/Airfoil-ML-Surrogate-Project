import os
import pandas as pd
import numpy as np


# ========================================
# GENERATE AIRFOIL DATASET
# ========================================
def generate_airfoil_dataset(n_samples=5000):

    # Random seed
    np.random.seed(42)

    # Airfoil types
    airfoils = [
        "NACA0012",
        "NACA2412",
        "NACA4412",
        "NACA6409"
    ]

    # Generate inputs
    data = {
        "airfoil": np.random.choice(
            airfoils,
            n_samples
        ),

        "angle_of_attack": np.random.uniform(
            -5,
            15,
            n_samples
        ),

        "reynolds_number": np.random.uniform(
            1e5,
            5e6,
            n_samples
        ),

        "velocity": np.random.uniform(
            20,
            250,
            n_samples
        )
    }

    # Create dataframe
    df = pd.DataFrame(data)

    # ========================================
    # Simulated Lift Coefficient
    # ========================================
    df["Cl"] = (
        0.1 * df["angle_of_attack"]
        + np.random.normal(
            0,
            0.05,
            n_samples
        )
    )

    # ========================================
    # Simulated Drag Coefficient
    # ========================================
    df["Cd"] = (
        0.01
        + 0.002 * (
            df["angle_of_attack"] ** 2
        )
        + np.random.normal(
            0,
            0.002,
            n_samples
        )
    )

    return df


# ========================================
# MAIN EXECUTION
# ========================================
if __name__ == "__main__":

    print("Generating airfoil dataset...")

    # Generate dataset
    df = generate_airfoil_dataset()

    # ========================================
    # PROJECT PATHS
    # ========================================
    BASE_DIR = os.path.dirname(
        os.path.abspath(__file__)
    )

    DATASET_DIR = os.path.join(
        BASE_DIR,
        "../dataset"
    )

    # Create folder if missing
    os.makedirs(
        DATASET_DIR,
        exist_ok=True
    )

    # Final CSV path
    DATASET_PATH = os.path.join(
        DATASET_DIR,
        "airfoil_data.csv"
    )

    # ========================================
    # SAVE DATASET
    # ========================================
    df.to_csv(
        DATASET_PATH,
        index=False
    )

    # ========================================
    # SUCCESS OUTPUT
    # ========================================
    print("\nDataset generated successfully!")

    print(f"\nDataset saved at:\n{DATASET_PATH}")

    print("\nDataset shape:")
    print(df.shape)

    print("\nFirst 5 rows:")
    print(df.head())