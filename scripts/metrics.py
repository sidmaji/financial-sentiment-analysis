import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def per_source_metrics(y_true, y_pred, sources):
    # Convert inputs to pandas Series to allow boolean indexing
    y_true = pd.Series(y_true).reset_index(drop=True)
    y_pred = pd.Series(y_pred).reset_index(drop=True)
    sources = pd.Series(sources).reset_index(drop=True)

    results = []
    for source_name in sources.unique():
        idx = sources == source_name
        acc = accuracy_score(y_true[idx], y_pred[idx])
        f1 = f1_score(y_true[idx], y_pred[idx], average="macro")
        results.append((source_name, acc, f1))
    return results
