# NEO Hazard Prediction

Classifies whether a Near-Earth Object (asteroid) is hazardous based on its
physical and orbital characteristics (diameter, relative velocity, miss
distance, absolute magnitude).

## Project structure

```
neo_hazard_project/
├── data/
│   └── neo_messy_balanced.csv     # raw dataset
├── src/
│   ├── preprocessing.py           # shared cleaning logic (used by both scripts below)
│   ├── train_compare.py           # trains & tunes KNN, DT, NB, RF, LR — prints ranked comparison
│   ├── train_final.py             # builds the ONE deployable Pipeline(scaler + model), saves .pkl
│   ├── llm_explainer.py           # Gemini call that explains a prediction in plain language
│   └── app.py                     # Streamlit demo app
├── models/                        # generated: pipeline .pkl, feature list, comparison results
├── requirements.txt
└── README.md
```

## Why it's split this way

- **preprocessing.py** — one source of truth for cleaning. Both training
  scripts import it, so they can never drift out of sync with each other.
- **train_compare.py** — the "which algorithm wins" experiment. Run this
  first, look at the ranked table, decide on a model.
- **train_final.py** — once you've picked a winner, hardcode it here (see
  the `FINAL_MODEL` variable at the top) and run this to produce the actual
  deployable artifact. Scaling is bundled INSIDE the pipeline object here —
  not applied separately — so the saved `.pkl` is self-contained and can't
  be fed unscaled or wrongly-scaled input by mistake.
- **llm_explainer.py** — isolated so the LLM piece can fail without taking
  down predictions, and so the API key lives in an environment variable
  instead of being hardcoded in code that might get shared.
- **app.py** — thin UI layer. Only talks to the saved pipeline and the LLM
  wrapper; contains no modeling logic of its own.

## Setup

```bash
pip install -r requirements.txt
export GEMINI_API_KEY="your-key-here"   # required only for the LLM explanation feature
```

## Usage

Run each step from inside `src/`:

```bash
cd src

# 1. Compare all 5 models, see which wins
python train_compare.py

# 2. Build & save the final deployable pipeline (edit FINAL_MODEL first if needed)
python train_final.py

# 3. Launch the demo app
streamlit run app.py
```

## Notes

- If you previously had a Gemini API key hardcoded in a notebook, revoke it
  in Google AI Studio and generate a new one — treat any key that was ever
  hardcoded as compromised.
- `train_compare.py` results are saved to `models/model_comparison_results.csv`
  so you can drop the table straight into a report/README/portfolio writeup.
