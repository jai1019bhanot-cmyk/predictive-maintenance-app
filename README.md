# Predictive Maintenance — Machine Failure Predictor

A Streamlit web app that predicts whether an industrial machine is at
risk of failure based on live sensor readings, using a Random Forest
Classifier trained on the AI4I 2020 Predictive Maintenance Dataset.

## Project structure

```
predictive_maintenance_app/
├── app.py                    # Streamlit app (Home, About, Prediction Tool)
├── random_forest_model.pkl   # Trained model (you provide this — see below)
├── requirements.txt          # Python dependencies
└── retrain_model_colab.py     # Script to retrain a clean model in Colab
```

## Step 1 — Retrain the model in Colab

Your previous model file had 10,012 features because the `Product ID`
column was one-hot encoded (creating ~10,000 dummy columns), and it
also included the row-index `UDI` and the failure-mode flag columns
(`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) as inputs — these are only known
*after* a failure happens, so they're data leakage.

Open `retrain_model_colab.py` in Colab, update the path to your
`ai4i2020.csv` file, and run it. It trains a clean Random Forest on
just 6 real-world inputs:

- Air temperature [K]
- Process temperature [K]
- Rotational speed [rpm]
- Torque [Nm]
- Tool wear [min]
- Type (L / M / H)

It will download `random_forest_model.pkl` to your computer.

## Step 2 — Run locally

Place the new `random_forest_model.pkl` in this same folder as
`app.py`, then:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Step 3 — Deploy to Streamlit Community Cloud

1. Push this folder (including `app.py`, `requirements.txt`, and
   `random_forest_model.pkl`) to a GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in
   with GitHub.
3. Click "New app", select your repo, branch, and set the main file
   path to `app.py`.
4. Click "Deploy". Streamlit Cloud will install the requirements and
   launch the app automatically.

## Notes

- The model is loaded with `joblib.load()`, not `pickle.load()` —
  this matters because scikit-learn models saved via `joblib.dump()`
  can fail with a cryptic `UnpicklingError` if loaded with plain
  `pickle`.
- `@st.cache_resource` ensures the model is loaded once and reused
  across user interactions, keeping the app responsive.
