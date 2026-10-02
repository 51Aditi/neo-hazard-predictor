"""
llm_explainer.py
-----------------
Wraps the Gemini call that explains a model's prediction in plain language.
Kept separate so the LLM integration can fail/degrade independently without
touching model logic, and so the API key never sits inside a notebook.

IMPORTANT: set your key as an environment variable, never hardcode it:
    export GEMINI_API_KEY="your-key-here"      (mac/linux)
    setx GEMINI_API_KEY "your-key-here"        (windows)

If you previously had a key hardcoded in a notebook that you shared/pushed
anywhere (GitHub, etc.), revoke it in Google AI Studio and generate a new one.
"""

import os

from google import genai

MODEL_NAME = "gemini-3.8-flash" # update to whatever current model you're using


def _get_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable not set. "
            "Set it before running the app (see module docstring)."
        )
    return genai.Client(api_key=api_key)


def explain_prediction(result: str, feature_values: dict) -> str:
    """
    result: "Hazardous" or "Not Hazardous"
    feature_values: dict of {feature_name: value} for the asteroid predicted on
    Returns a plain-language explanation string. Falls back to a canned
    message if the LLM call fails, so the app never breaks on an API hiccup.
    """
    prompt = f"""
You are an asteroid hazard prediction assistant.

The machine learning model predicted: {result}

Asteroid data:
{feature_values}

Explain this prediction in simple, non-technical language for a general
audience. Do not change or contradict the machine learning prediction.
Do not invent reasons that aren't supported by the data provided.
Keep it to 2-3 sentences.
"""
    try:
        client = _get_client()
        response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
        return response.text
    except Exception as e:
        return (
            f"(LLM explanation unavailable: {e}) "
            f"Model prediction stands as: {result}."
        )
