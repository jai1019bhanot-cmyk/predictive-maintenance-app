import streamlit as st
import pandas as pd
import joblib
import os

# ------------------------------------------------------------
# Page config
# ------------------------------------------------------------
st.set_page_config(
    page_title="MFPS · Machine Failure Prediction",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------
# Custom CSS — Industrial Dashboard Theme
# ------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ── Base ── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background-color: #080a0f;
        color: #c9d1d9;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background-color: #0d1117;
        border-right: 1px solid #1c2333;
    }

    section[data-testid="stSidebar"] .stRadio label {
        color: #8b949e !important;
        font-family: 'Inter', sans-serif;
        font-size: 0.9rem;
        transition: color 0.2s;
    }

    section[data-testid="stSidebar"] .stRadio label:hover {
        color: #f59e0b !important;
    }

    /* ── Headings ── */
    h1, h2, h3, h4 {
        font-family: 'Space Grotesk', sans-serif !important;
        color: #ffffff !important;
        letter-spacing: -0.02em;
    }

    h1 { font-size: 2.4rem !important; font-weight: 700 !important; }
    h2 { font-size: 1.6rem !important; font-weight: 600 !important; }
    h3 { font-size: 1.2rem !important; font-weight: 600 !important; }

    /* ── Paragraphs and markdown ── */
    p, .stMarkdown p {
        color: #8b949e;
        line-height: 1.7;
        font-size: 0.95rem;
    }

    /* ── Inputs ── */
    .stNumberInput input, .stSelectbox select {
        background-color: #161b22 !important;
        border: 1px solid #2a2f3a !important;
        border-radius: 8px !important;
        color: #e6edf3 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.9rem !important;
    }

    .stNumberInput input:focus, .stSelectbox select:focus {
        border-color: #f59e0b !important;
        box-shadow: 0 0 0 2px rgba(245, 158, 11, 0.15) !important;
    }

    label {
        color: #8b949e !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.05em !important;
        text-transform: uppercase !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* ── Button ── */
    .stButton > button {
        background: linear-gradient(135deg, #f59e0b, #d97706) !important;
        color: #000000 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.65rem 2rem !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 700 !important;
        font-size: 0.9rem !important;
        letter-spacing: 0.04em !important;
        text-transform: uppercase !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 0 20px rgba(245, 158, 11, 0.3) !important;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #fbbf24, #f59e0b) !important;
        box-shadow: 0 0 30px rgba(245, 158, 11, 0.5) !important;
        transform: translateY(-1px) !important;
    }

    /* ── Table ── */
    .stDataFrame, .stTable {
        background-color: #0d1117 !important;
        border: 1px solid #1c2333 !important;
        border-radius: 10px !important;
    }

    thead tr th {
        background-color: #161b22 !important;
        color: #f59e0b !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 0.82rem !important;
        letter-spacing: 0.05em !important;
        text-transform: uppercase !important;
        border-bottom: 1px solid #2a2f3a !important;
    }

    tbody tr td {
        color: #c9d1d9 !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.88rem !important;
        border-bottom: 1px solid #1c2333 !important;
    }

    tbody tr:hover td {
        background-color: #161b22 !important;
    }

    /* ── Divider ── */
    hr {
        border-color: #1c2333 !important;
        margin: 2rem 0 !important;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #080a0f; }
    ::-webkit-scrollbar-thumb { background: #2a2f3a; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #f59e0b; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="padding: 1rem 0 1.5rem 0;">
            <div style="font-family: 'Space Grotesk', sans-serif; font-size: 1.1rem;
                        font-weight: 700; color: #f59e0b; letter-spacing: -0.01em;">
                ⚡ MFPS
            </div>
            <div style="font-family: 'Inter', sans-serif; font-size: 0.72rem;
                        color: #4a5568; letter-spacing: 0.1em; text-transform: uppercase;
                        margin-top: 2px;">
                Machine Failure Prediction System
            </div>
        </div>
        <hr style="border-color: #1c2333; margin: 0 0 1.2rem 0;">
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigate",
        ["Home", "About the Project", "Prediction Tool"],
        label_visibility="collapsed",
    )

    st.markdown(
        """
        <div style="position: absolute; bottom: 2rem; left: 1rem; right: 1rem;">
            <hr style="border-color: #1c2333;">
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.7rem;
                        color: #2a2f3a; line-height: 1.6;">
                MODEL: Random Forest<br>
                DATASET: AI4I 2020<br>
                FEATURES: 7
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# HOME PAGE
# ============================================================
if page == "Home":

    # Hero
    st.markdown(
        """
        <div style="padding: 2.5rem 0 1rem 0;">
            <div style="font-family: 'Inter', sans-serif; font-size: 0.75rem;
                        color: #f59e0b; letter-spacing: 0.15em; text-transform: uppercase;
                        margin-bottom: 0.8rem;">
                ● SYSTEM ONLINE
            </div>
            <h1 style="font-family: 'Space Grotesk', sans-serif; font-size: 3rem;
                       font-weight: 700; color: #ffffff; line-height: 1.1;
                       letter-spacing: -0.03em; margin: 0 0 1rem 0;">
                Industrial Machine<br>
                <span style="color: #f59e0b;">Failure Prediction</span>
            </h1>
            <p style="font-family: 'Inter', sans-serif; font-size: 1rem;
                      color: #6b7280; max-width: 560px; line-height: 1.7; margin: 0;">
                A Random Forest classifier trained on the AI4I 2020 dataset.
                Feed it live sensor readings — get an instant failure risk verdict.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Three cards
    col1, col2, col3 = st.columns(3)

    cards = [
        ("01", "INPUT", "#f59e0b",
         "Five real-time sensor readings plus machine type (L / M / H) are passed to the model."),
        ("02", "MODEL", "#f59e0b",
         "Random Forest trained to detect failure-preceding sensor patterns in 10 000+ industrial cycles."),
        ("03", "OUTPUT", "#f59e0b",
         "Binary verdict — Normal or Failure Risk — with failure probability score."),
    ]

    for col, (num, title, accent, desc) in zip([col1, col2, col3], cards):
        with col:
            st.markdown(
                f"""
                <div style="background: #0d1117; border: 1px solid #1c2333;
                            border-top: 2px solid {accent}; border-radius: 10px;
                            padding: 1.5rem; height: 100%;">
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.65rem;
                                color: {accent}; letter-spacing: 0.1em; margin-bottom: 0.6rem;">
                        {num}
                    </div>
                    <div style="font-family: 'Space Grotesk', sans-serif; font-size: 1rem;
                                font-weight: 600; color: #ffffff; margin-bottom: 0.7rem;
                                letter-spacing: 0.03em;">
                        {title}
                    </div>
                    <div style="font-family: 'Inter', sans-serif; font-size: 0.85rem;
                                color: #6b7280; line-height: 1.6;">
                        {desc}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br><hr>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color: #4a5568; font-size: 0.85rem;'>"
        "Use the sidebar to explore project details or go straight to the "
        "<strong style='color: #f59e0b;'>Prediction Tool</strong>.</p>",
        unsafe_allow_html=True,
    )


# ============================================================
# ABOUT PAGE
# ============================================================
elif page == "About the Project":

    st.markdown(
        """
        <div style="padding: 2rem 0 1.5rem 0;">
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.72rem;
                        color: #f59e0b; letter-spacing: 0.12em; margin-bottom: 0.7rem;">
                /ABOUT
            </div>
            <h1 style="font-family: 'Space Grotesk', sans-serif; font-weight: 700;
                       letter-spacing: -0.03em; margin: 0;">About the Project</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Dataset")
    st.markdown(
        "This app runs on the **AI4I 2020 Predictive Maintenance Dataset** — a synthetic "
        "dataset mirroring real industrial sensor logs. Each record captures one operational "
        "cycle: temperature readings, rotational speed, torque, tool wear, product type, and "
        "whether failure occurred."
    )

    st.markdown("<br>", unsafe_allow_html=True)
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
                "Temperature of the active manufacturing process",
                "Tool/spindle speed in revolutions per minute",
                "Torque applied during the operation",
                "Cumulative wear time on the current tool",
                "Product quality tier: Low, Medium, or High",
            ],
        }
    )
    st.table(feature_table)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Why these features — not others")
    st.markdown(
        "The original dataset also includes `Product ID`, `UDI`, and individual failure-mode "
        "flags (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`). All were deliberately excluded. "
        "The identifiers carry no predictive signal, and the failure-mode flags are only "
        "observable *after* a failure — using them would be data leakage, inflating accuracy "
        "to meaningless levels."
    )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("### Tech stack")

    tech = [
        ("Python", "Data processing & model training"),
        ("pandas", "Data wrangling & feature engineering"),
        ("scikit-learn", "Random Forest Classifier (100 estimators)"),
        ("Streamlit", "Interactive web interface"),
        ("joblib", "Model serialization"),
    ]

    cols = st.columns(len(tech))
    for col, (name, desc) in zip(cols, tech):
        with col:
            st.markdown(
                f"""
                <div style="background: #0d1117; border: 1px solid #1c2333;
                            border-radius: 8px; padding: 1rem; text-align: center;">
                    <div style="font-family: 'Space Grotesk', sans-serif; font-weight: 600;
                                color: #f59e0b; font-size: 0.85rem; margin-bottom: 0.4rem;">
                        {name}
                    </div>
                    <div style="font-family: 'Inter', sans-serif; font-size: 0.75rem;
                                color: #4a5568; line-height: 1.5;">
                        {desc}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================
# PREDICTION TOOL PAGE
# ============================================================
elif page == "Prediction Tool":

    st.markdown(
        """
        <div style="padding: 2rem 0 1.5rem 0;">
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.72rem;
                        color: #f59e0b; letter-spacing: 0.12em; margin-bottom: 0.7rem;">
                /PREDICT
            </div>
            <h1 style="font-family: 'Space Grotesk', sans-serif; font-weight: 700;
                       letter-spacing: -0.03em; margin: 0;">Prediction Tool</h1>
            <p style="color: #6b7280; font-size: 0.92rem; margin-top: 0.6rem;">
                Enter current sensor readings. Hit <strong style="color:#f59e0b;">RUN ANALYSIS</strong> for a failure risk verdict.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        model = load_model()
        model_loaded = True
    except FileNotFoundError:
        model_loaded = False
        st.markdown(
            """
            <div style="background: #1a0a0a; border: 1px solid #7f1d1d; border-radius: 10px;
                        padding: 1.2rem 1.5rem; color: #f87171; font-family: 'Inter', sans-serif;
                        font-size: 0.88rem;">
                ⚠ Model file not found. Ensure <code>random_forest_model.pkl</code>
                is in the same directory as <code>app.py</code>.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Input section header
    st.markdown(
        """
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.7rem;
                    color: #2a2f3a; letter-spacing: 0.1em; text-transform: uppercase;
                    margin-bottom: 1rem; padding-top: 0.5rem;
                    border-top: 1px solid #1c2333;">
            ── SENSOR INPUTS
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2, gap="large")

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
        machine_type = st.selectbox("Machine type", ["L — Low", "M — Medium", "H — High"])

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("⚡  RUN ANALYSIS"):
        if not model_loaded:
            st.warning("Cannot predict — model file is missing.")
        else:
            machine_type_code = machine_type[0]
            type_l = 1 if machine_type_code == "L" else 0
            type_m = 1 if machine_type_code == "M" else 0

            input_df = pd.DataFrame(
                [[air_temp, process_temp, rotational_speed, torque, tool_wear, type_l, type_m]],
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

            st.markdown("<br>", unsafe_allow_html=True)

            if prediction == 1:
                st.markdown(
                    f"""
                    <style>
                    @keyframes alarm-pulse {{
                        0% {{ box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.6); }}
                        70% {{ box-shadow: 0 0 0 18px rgba(239, 68, 68, 0); }}
                        100% {{ box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }}
                    }}
                    .result-fail {{
                        animation: alarm-pulse 1.5s infinite;
                    }}
                    </style>
                    <div class="result-fail" style="background: #0f0505; border: 1px solid #ef4444;
                                border-left: 4px solid #ef4444; border-radius: 10px;
                                padding: 1.8rem 2rem;">
                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.65rem;
                                    color: #ef4444; letter-spacing: 0.15em; margin-bottom: 0.5rem;">
                            VERDICT
                        </div>
                        <div style="font-family: 'Space Grotesk', sans-serif; font-size: 1.8rem;
                                    font-weight: 700; color: #ef4444; letter-spacing: -0.02em;">
                            ⚠ FAILURE RISK DETECTED
                        </div>
                        <div style="font-family: 'Inter', sans-serif; font-size: 0.9rem;
                                    color: #9b1c1c; margin-top: 0.5rem;">
                            Estimated failure probability:
                            <span style="font-family: 'JetBrains Mono', monospace;
                                         color: #ef4444; font-size: 1.1rem; font-weight: 600;">
                                {probability:.1%}
                            </span>
                        </div>
                        <div style="font-family: 'Inter', sans-serif; font-size: 0.8rem;
                                    color: #4a5568; margin-top: 1rem;">
                            Recommend immediate inspection of high-wear components.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            
            else:
                st.markdown(
                    f"""
                    <style>
                    @keyframes ok-pulse {{
                        0% {{ box-shadow: 0 0 0 0 rgba(34, 197, 94, 0.35); }}
                        70% {{ box-shadow: 0 0 0 14px rgba(34, 197, 94, 0); }}
                        100% {{ box-shadow: 0 0 0 0 rgba(34, 197, 94, 0); }}
                    }}
                    .result-ok {{
                        animation: ok-pulse 2.5s infinite;
                    }}
                    </style>
                    <div class="result-ok" style="background: #050f0a; border: 1px solid #22c55e;
                                border-left: 4px solid #22c55e; border-radius: 10px;
                                padding: 1.8rem 2rem;">
                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.65rem;
                                    color: #22c55e; letter-spacing: 0.15em; margin-bottom: 0.5rem;">
                            STATUS / VERIFIED
                        </div>
                        <div style="font-family: 'Space Grotesk', sans-serif; font-size: 1.8rem;
                                    font-weight: 700; color: #22c55e; letter-spacing: -0.02em;">
                            ✓ MACHINE HEALTHY
                        </div>
                        <div style="font-family: 'Inter', sans-serif; font-size: 0.9rem;
                                    color: #15803d; margin-top: 0.5rem;">
                            Failure probability:
                            <span style="font-family: 'JetBrains Mono', monospace;
                                         color: #22c55e; font-size: 1.1rem; font-weight: 600;">
                                {probability:.1%}
                            </span>
                        </div>
                        <div style="font-family: 'Inter', sans-serif; font-size: 0.8rem;
                                    color: #4a5568; margin-top: 1rem;">
                            All sensor readings indicate stable machine operation.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                """
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.7rem;
                            color: #2a2f3a; letter-spacing: 0.1em; padding-top: 1rem;
                            border-top: 1px solid #1c2333; margin-bottom: 0.8rem;">
                    ── INPUT SUMMARY
                </div>
                """,
                unsafe_allow_html=True,
            )
            summary_cols = st.columns(4)
            summary_data = [
                ("Air Temp", f"{air_temp} K"),
                ("Process Temp", f"{process_temp} K"),
                ("Rot. Speed", f"{rotational_speed} rpm"),
                ("Torque", f"{torque} Nm"),
                ("Tool Wear", f"{tool_wear} min"),
                ("Type", machine_type_code),
                ("Failure Prob", f"{probability:.3f}"),
            ]
            for i, (label, val) in enumerate(summary_data):
                with summary_cols[i % 4]:
                    st.markdown(
                        f"""
                        <div style="background: #0d1117; border: 1px solid #1c2333;
                                    border-radius: 6px; padding: 0.7rem 0.9rem; margin-bottom: 0.5rem;">
                            <div style="font-family: 'Inter', sans-serif; font-size: 0.65rem;
                                        color: #4a5568; text-transform: uppercase; letter-spacing: 0.08em;">
                                {label}
                            </div>
                            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.9rem;
                                        color: #f59e0b; font-weight: 500; margin-top: 0.2rem;">
                                {val}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
