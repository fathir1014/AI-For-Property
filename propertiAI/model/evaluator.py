from sklearn.metrics import (
  accuracy_score, precision_score, recall_score, f1_score,
  confusion_matrix, classification_report, mean_squared_error,
  mean_absolute_error, r2_score, roc_auc_score, roc_curve
)
import numpy as np

def evaluate_model_classifier(y_true, y_pred, y_prob=None):
    result = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1 Score': f1_score(y_true, y_pred, zero_division=0),
        'Confusion Matrix': confusion_matrix(y_true, y_pred),
        'Report (dict)': classification_report(y_true, y_pred, output_dict=True),
        'Report (text)': classification_report(y_true, y_pred),
    }

    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_score = roc_auc_score(y_true, y_prob)
        result['FPR'] = fpr
        result['TPR'] = tpr
        result['AUC Score'] = auc_score

    return result

def evaluate_model_regression(name, y_true, y_pred) :
    return {
        'MSE' : mean_squared_error(y_true, y_pred),
        'MAE' : mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2 Score' : r2_score(y_true, y_pred)
    }