import matplotlib.pyplot as plt
import seaborn as sns
import os
from config.settings import BASE_DIR

def plot_confusion_matrix(cm, model_name="Model", save=False, filename='Confusion_matrix.png'):

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    if save:
        plt.savefig(f"{BASE_DIR}/output/{filename}")
        print(f"[INFO] Saved: {BASE_DIR}/output/{filename}")
    else:
        plt.show()

def plot_roc_curve(fpr, tpr, auc_score, model_name="Model", save=False, filename='roc_curve.png'):

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
    plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    if save:
        plt.savefig(f"{BASE_DIR}/output/{filename}")
        print(f"[INFO] Saved: {BASE_DIR}/output/{filename}")
    else:
        plt.show()
