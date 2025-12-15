# Load the hugging face model

import streamlit as st
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline
)
from streamlit_extras.star_rating import star_rating

@st.cache_resource
def load_model():
    MODEL_NAME = "andrewoh/RoBERTa-finetuned-hotel-reviews-sentiment-analysis"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    sentiment_pipeline = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        truncation=True,
        max_length=512,
        return_all_scores=True
    )
    return sentiment_pipeline


# def load_model():
#     return pipeline(
#         task="sentiment-analysis",
#         model="andrewoh/RoBERTa-finetuned-hotel-reviews-sentiment-analysis"
#     )


# ----------------------------
# Scoring function
# ----------------------------
def calc_review_sentiment_score(review_results):
    label_multipliers = {
        "LABEL_0": 0.0,
        "LABEL_1": 0.25,
        "LABEL_2": 0.5,
        "LABEL_3": 0.75,
        "LABEL_4": 1.0
    }

    score = sum(
        label["score"] * label_multipliers.get(label["label"], 0.0)
        for label in review_results
    )
    return score * 100

# ----------------------------
# Other Helper UI functions
# ----------------------------

def score_to_stars(score):
    # Convert 0–100 score → 0–5 stars (rounded to nearest 0.1)
    return round(score/20, 1)

def render_stars(star_rating):

    full = int(star_rating)
    half = 1 if star_rating - full >= 0.5 else 0
    empty = 5 - full - half
    return "⭐" * full + "⭐️" * half + "☆" * empty