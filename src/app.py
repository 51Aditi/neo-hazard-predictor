"""
app.py
------
Streamlit demo app. Loads the pipeline saved by train_final.py (scaler +
model bundled together, so no manual scaling needed here) and serves an
interactive hazard prediction UI, optionally explained by the LLM layer.

Run:
    streamlit run src/app.py
"""

import json
import pickle

import numpy as np
import pandas as pd
import streamlit as st

from llm_explainer import explain_prediction

MODEL_PATH = "models/neo_hazard_pipeline.pkl"
FEATURE_LIST_PATH = "models/feature_columns.json"


@st.cache_resource
def load_pipeline():
    with open(MODEL_PATH, "rb") as f:
        pipeline = pickle.load(f)
    with open(FEATURE_LIST_PATH) as f:
        feature_columns = json.load(f)
    return pipeline, feature_columns


def main():
    st.set_page_config(page_title="NEO Hazard Predictor", page_icon="☄️")
    st.title("☄️ Near-Earth Object Hazard Predictor")
    st.write(
        "Enter an asteroid's physical and orbital characteristics to predict "
        "whether it's classified as hazardous."
    )

    try:
        pipeline, feature_columns = load_pipeline()
    except FileNotFoundError:
        st.error(
            "Model files not found. Run `python train_final.py` from the "
            "`src/` folder first to generate the pipeline."
        )
        return

    st.subheader("Asteroid Details")
    col1, col2 = st.columns(2)

    inputs = {}
    with col1:
        inputs["est_diameter_min"] = st.number_input("Est. diameter min (km)", min_value=0.0, value=0.1, format="%.4f")
        inputs["est_diameter_max"] = st.number_input("Est. diameter max (km)", min_value=0.0, value=0.5, format="%.4f")
        inputs["relative_velocity"] = st.number_input("Relative velocity (km/h)", min_value=0.0, value=25000.0)
    with col2:
        inputs["miss_distance"] = st.number_input("Miss distance (km)", min_value=0.0, value=5000000.0)
        inputs["absolute_magnitude"] = st.number_input("Absolute magnitude", min_value=0.0, value=20.0)

    # Build input row in the exact column order the model was trained on
    missing = [c for c in feature_columns if c not in inputs]
    if missing:
        st.error(f"App inputs don't match model's expected features: missing {missing}")
        return

    input_df = pd.DataFrame([[inputs[c] for c in feature_columns]], columns=feature_columns)

    if st.button("Predict Hazard Status", type="primary"):
        prediction = pipeline.predict(input_df)[0]
        result = "Hazardous" if prediction == 1 else "Not Hazardous"

        if result == "Hazardous":
            st.error(f"⚠️ Prediction: **{result}**")
        else:
            st.success(f"✅ Prediction: **{result}**")

        # Show prediction probability if the model supports it
        if hasattr(pipeline.named_steps["model"], "predict_proba"):
            proba = pipeline.predict_proba(input_df)[0]
            st.write(f"Confidence — Not Hazardous: {proba[0]:.1%}, Hazardous: {proba[1]:.1%}")

        with st.spinner("Getting plain-language explanation..."):
            explanation = explain_prediction(result, inputs)
        st.info(explanation)


if __name__ == "__main__":
    main()
