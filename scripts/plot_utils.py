import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(
    y_true, y_pred, labels=None, title="Confusion Matrix", cmap="Blues"
):
    # Plots a labeled confusion matrix.
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(5, 4), dpi=125)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap=cmap, xticklabels=labels, yticklabels=labels
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.show()
