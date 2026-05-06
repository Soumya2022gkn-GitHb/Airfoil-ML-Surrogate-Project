import os
import pandas as pd
import numpy as np


# ========================================
# SIMULATED XFOIL CFD RUNNER
# ========================================
def run_xfoil_simulation(airfoil, angle_of_attack, reynolds_number):
    """
    Simulate aerodynamic coefficients.

    In a real production system:
    - This function would call XFoil/OpenFOAM
    - Run CFD simulations
    - Extract aerodynamic coefficients

    Here we generate realistic synthetic outputs.
    """

    # Simulated Lift Coefficient (Cl)
    cl = (
        0.1 * angle_of_attack
        + np.random.normal(0, 0.02)
    )

    # Simulated Drag Coefficient (Cd)
    cd = (
        0.01
        + 0.002 * (angle_of_attack ** 2)
        + np.random.normal(0, 0.001)
    )

    return cl, cd


# ========================================
# GENERATE CFD DATASET
# ========================================
def generate_cfd_dataset(n_samples=1000):

    np.random.seed(42)

    airfoils = [
        "NACA0012",
        "NACA2412",
        "NACA4412",
        "NACA6409"
    ]

    dataset = []

    for _ in range(n_samples):

        airfoil = np.random.choice(airfoils)

        angle_of_attack = np.random.uniform(
            -5,
            15
        )

        reynolds_number = np.random.uniform(
            1e5,
            5e6
        )

        velocity = np.random.uniform(
            20,
            250
        )

        # Simulated CFD run
        cl, cd = run_xfoil_simulation(
            airfoil,
            angle_of_attack,
            reynolds_number
        )

        dataset.append({
            "airfoil": airfoil,
            "angle_of_attack": angle_of_attack,
            "reynolds_number": reynolds_number,
            "velocity": velocity,
            "Cl": cl,
            "Cd": cd
        })

    return pd.DataFrame(dataset)


# ========================================
# MAIN EXECUTION
# ========================================
if __name__ == "__main__":

    print("Starting XFoil CFD simulation...")

    # Generate synthetic CFD dataset
    df = generate_cfd_dataset()

    # Create dataset directory
    os.makedirs(
        "../dataset",
        exist_ok=True
    )

    # Save dataset
    output_path = "../dataset/airfoil_data.csv"

    df.to_csv(
        output_path,
        index=False
    )

    print(f"CFD dataset saved to: {output_path}")

    print("XFoil simulation completed successfully!")
