import streamlit as st
import pandas as pd
import joblib
import os

# ------------------------------------------------------------
# Page config
# ------------------------------------------------------------
st.set_page_config(
    page_title="Predictive Maintenance | Machine Failure Predictor",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------
# Dark theme styling
# ------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0e1117;
        color: #e6e6e6;
    }
    section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #2a2f3a;
    }
    h1, h2, h3 {
        color: #f5f5f5;
        font-weight: 600;
    }
    .stButton > button {
        background-color: #2563eb;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.6rem 1.4rem;
        font-weight: 600;
        transition: background-color 0.2s ease;
    }
    .stButton > button:hover {
        background-color: #1d4ed8;
        color: white;
    }
    .metric-card {
        background-color: #161b22;
        border: 1px solid #2a2f3a;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
    }
    .result-ok {
        background-color: #052e1c;
        border: 1px solid #15803d;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        color: #4ade80;
        font-size: 1.1rem;
        font-weight: 600;
    }
    .result-fail {
        background-color: #2e0505;
        border: 1px solid #b91c1c;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        color: #f87171;
        font-size: 1.1rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# Load model (cached so it's only loaded once)
# ------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


# ------------------------------------------------------------
# Sidebar navigation
# ------------------------------------------------------------
st.sidebar.title("⚙️ Predictive Maintenance")
page = st.sidebar.radio(
    "Navigate",
    ["Home", "About the Project", "Prediction Tool"],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<span style='color:#8b949e; font-size:0.85rem;'>"
    "Built with scikit-learn &amp; Streamlit"
    "</span>",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# HOME PAGE
# ------------------------------------------------------------
if page == "Home":
    st.title("Industrial Machine Failure Prediction")
    st.markdown(
        "A machine learning system that predicts the likelihood of "
        "machine failure from real-time sensor readings, trained on the "
        "**AI4I 2020 Predictive Maintenance Dataset**."
    )

    st.markdown("### What this app does")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            "<div class='metric-card'><h3>📊 Input</h3>"
            "Five live sensor readings plus the machine's product type "
            "(L / M / H) are fed into the model.</div>",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            "<div class='metric-card'><h3>🌲 Model</h3>"
            "A Random Forest Classifier trained to detect patterns that "
            "precede equipment failure.</div>",
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            "<div class='metric-card'><h3>⚡ Output</h3>"
            "A binary prediction — Normal Operation or Failure Risk — "
            "delivered instantly.</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown(
        "Use the sidebar to explore the project details or jump straight "
        "into the **Prediction Tool** to test it with your own values."
    )

# ------------------------------------------------------------
# ABOUT PAGE
# ------------------------------------------------------------
elif page == "About the Project":
    st.title("About the Project")

    st.markdown("### Dataset")
    st.markdown(
        "This project uses the **AI4I 2020 Predictive Maintenance "
        "Dataset**, a synthetic dataset that mirrors real industrial "
        "sensor data. Each record represents one operational cycle of a "
        "machine, capturing temperature, rotational speed, torque, tool "
        "wear, and the product type, along with whether the machine "
        "experienced a failure during that cycle."
    )

    st.markdown("### Features used by the model")
    feature_table = pd.DataFrame(
        {
            "Feature": [
                "Air temperature [K]",
                "Process temperature [K]",
                "Rotational speed [rpm]",
                "Torque [Nm]",
                "Tool wear [min]",
                "Type",
            ],
            "Description": [
                "Ambient air temperature around the machine",
                "Temperature of the manufacturing process",
                "Speed of the tool/spindle in revolutions per minute",
                "Torque applied during the operation",
                "Cumulative wear time on the current tool",
                "Product quality variant: Low, Medium, or High",
            ],
        }
    )
    st.table(feature_table)

    st.markdown("### Model")
    st.markdown(
        "A **Random Forest Classifier** (100 estimators) was trained on "
        "these six inputs to predict the binary target `Machine failure`. "
        "Random forests work well for this kind of tabular sensor data "
        "because they capture non-linear interactions between variables "
        "(for example, how torque and tool wear combine to increase "
        "failure risk) without requiring heavy preprocessing."
    )

    st.markdown("### Why these features (and not others)")
    st.markdown(
        "The original dataset also contains a unique `Product ID`, a row "
        "identifier `UDI`, and individual failure-mode flags "
        "(`TWF`, `HDF`, `PWF`, `OSF`, `RNF`). These were deliberately "
        "excluded: `Product ID` and `UDI` are just identifiers with no "
        "predictive meaning, and the failure-mode flags are only known "
        "*after* a failure has occurred, so including them would be "
        "data leakage. Restricting the model to genuinely observable, "
        "real-time sensor data keeps it realistic for actual deployment."
    )

    st.markdown("### Tech stack")
    st.markdown(
        "- **Python** for data processing and model training\n"
        "- **pandas** for data manipulation\n"
        "- **scikit-learn** for the Random Forest model\n"
        "- **Streamlit** for the interactive web interface\n"
        "- **joblib** for model serialization"
    )

# ------------------------------------------------------------
# PREDICTION TOOL PAGE
# ------------------------------------------------------------
elif page == "Prediction Tool":
    st.title("Machine Failure Prediction Tool")
    st.markdown(
        "Enter the current sensor readings below and click **Predict** "
        "to check whether the machine is at risk of failure."
    )

    try:
        model = load_model()
        model_loaded = True
    except FileNotFoundError:
        model_loaded = False
        st.error(
            "Model file `random_forest_model.pkl` not found in the app "
            "directory. Make sure it's uploaded alongside `app.py`."
        )

    col1, col2 = st.columns(2)

    with col1:
        air_temp = st.number_input(
            "Air temperature [K]", min_value=290.0, max_value=310.0,
            value=298.0, step=0.1,
        )
        process_temp = st.number_input(
            "Process temperature [K]", min_value=300.0, max_value=320.0,
            value=308.0, step=0.1,
        )
        rotational_speed = st.number_input(
            "Rotational speed [rpm]", min_value=1000, max_value=3000,
            value=1500, step=10,
        )

    with col2:
        torque = st.number_input(
            "Torque [Nm]", min_value=0.0, max_value=80.0,
            value=40.0, step=0.1,
        )
        tool_wear = st.number_input(
            "Tool wear [min]", min_value=0, max_value=260,
            value=100, step=1,
        )
        machine_type = st.selectbox("Type", ["L", "M", "H"])

    st.markdown("")

    if st.button("Predict"):
        if not model_loaded:
            st.warning("Cannot predict — model file is missing.")
        else:
            # Build input row matching the training feature order:
            # ['Air temperature [K]', 'Process temperature [K]',
            #  'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]',
            #  'Type_L', 'Type_M']
            type_l = 1 if machine_type == "L" else 0
            type_m = 1 if machine_type == "M" else 0

            input_df = pd.DataFrame(
                [[
                    air_temp,
                    process_temp,
                    rotational_speed,
                    torque,
                    tool_wear,
                    type_l,
                    type_m,
                ]],
                columns=[
                    "Air temperature [K]",
                    "Process temperature [K]",
                    "Rotational speed [rpm]",
                    "Torque [Nm]",
                    "Tool wear [min]",
                    "Type_L",
                    "Type_M",
                ],
            )

            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][1]

            if prediction == 1:
                st.markdown(
                    f"<div class='result-fail'>⚠️ Failure Risk Detected "
                    f"— estimated probability: {probability:.1%}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div class='result-ok'>✅ Normal Operation "
                    f"— estimated failure probability: {probability:.1%}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
