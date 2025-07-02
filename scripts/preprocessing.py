import re

import pandas as pd


def clean_text(text):
    # Basic text cleaning: removes URLs, emojis, non-alphanumerics, and normalizes spacing.
    if pd.isnull(text):
        return ""
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
