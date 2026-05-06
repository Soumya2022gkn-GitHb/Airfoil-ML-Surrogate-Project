import os
import streamlit as st
import numpy as np
import joblib


# ========================================
# PAGE CONFIGURATION
# ========================================
st.set_page_config(
    page_title="Airfoil ML Surrogate",
    page_icon="✈️",
    layout="centered"
)


# ========================================
# PROJECT PATHS
# ========================================
BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "../models"
)


# ========================================
# LOAD TRAINED MODELS
# ========================================
cl_model = joblib.load(
    os.path.join(MODEL_DIR, "cl_model.pkl")
)

cd_model = joblib.load(
    os.path.join(MODEL_DIR, "cd_model.pkl")
)

encoder = joblib.load(
    os.path.join(MODEL_DIR, "label_encoder.pkl")
)


# ========================================
# APP TITLE
# ========================================
st.title("✈️ Real-Time Airfoil Lift & Drag Prediction")

st.markdown(
    """
    This Aero-CFD ML surrogate predicts:
    - Lift Coefficient (Cl)
    - Drag Coefficient (Cd)

    instantly using machine learning.
    """
)

st.divider()


# ========================================
# USER INPUT SECTION
# ========================================
st.subheader("Airfoil Configuration")

# Airfoil selection
airfoil = st.selectbox(
    "Select Airfoil",
    [
        "NACA0012",
        "NACA2412",
        "NACA4412",
        "NACA6409"
    ]
)

# Angle of attack
angle_of_attack = st.slider(
    "Angle of Attack (degrees)",
    min_value=-5.0,
    max_value=15.0,
    value=5.0,
    step=0.5
)

# Reynolds number
reynolds_number = st.number_input(
    "Reynolds Number",
    min_value=100000.0,
    max_value=5000000.0,
    value=1000000.0,
    step=100000.0
)

# Velocity
velocity = st.slider(
    "Velocity (m/s)",
    min_value=20.0,
    max_value=250.0,
    value=100.0,
    step=5.0
)

st.divider()


# ========================================
# PREDICTION BUTTON
# ========================================
if st.button(
    "Predict Aerodynamics",
    use_container_width=True
):

    # Encode airfoil type
    airfoil_encoded = encoder.transform(
        [airfoil]
    )[0]

    # Prepare feature array
    features = np.array([[
        airfoil_encoded,
        angle_of_attack,
        reynolds_number,
        velocity
    ]])

    # Predict lift coefficient
    predicted_cl = cl_model.predict(
        features
    )[0]

    # Predict drag coefficient
    predicted_cd = cd_model.predict(
        features
    )[0]

    # Compute Lift-to-Drag ratio
    lift_to_drag = predicted_cl / (
        predicted_cd + 1e-9
    )

    # ========================================
    # DISPLAY RESULTS
    # ========================================
    st.subheader("Prediction Results")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="Lift Coefficient (Cl)",
            value=f"{predicted_cl:.4f}"
        )

    with col2:
        st.metric(
            label="Drag Coefficient (Cd)",
            value=f"{predicted_cd:.5f}"
        )

    st.metric(
        label="Lift-to-Drag Ratio",
        value=f"{lift_to_drag:.2f}"
    )

    st.divider()

    # ========================================
    # AERODYNAMIC INTERPRETATION
    # ========================================
    st.subheader("Aerodynamic Interpretation")

    if lift_to_drag > 50:
        st.success(
            "Excellent aerodynamic efficiency."
        )

    elif lift_to_drag > 20:
        st.info(
            "Moderate aerodynamic efficiency."
        )

    else:
        st.warning(
            "Low aerodynamic efficiency."
        )

    # ========================================
    # SIMPLE PERFORMANCE BARS
    # ========================================
    st.subheader("Performance Indicators")

    st.write("Lift Coefficient")
    st.progress(
        min(float(abs(predicted_cl)) / 2.0, 1.0)
    )

    st.write("Drag Coefficient")
    st.progress(
        min(float(predicted_cd) / 0.2, 1.0)
    )

    st.write("Lift-to-Drag Efficiency")
    st.progress(
        min(float(lift_to_drag) / 100.0, 1.0)
    )


# ========================================
# FOOTER
# ========================================
st.divider()

st.caption(
    "Aero-CFD ML Surrogate System using Machine Learning and Streamlit"
)