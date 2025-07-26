import pandas as pd
import joblib
import os
import datetime
import numpy as np
from preprocessing.preprocessing import preprocessing
from preprocessing.clean_data import clean_data_classifier, clean_data_regression
from model.trainer import train_classifier, train_regressors
from model.evaluator import evaluate_model_classifier, evaluate_model_regression
from data.data_loader import load_data
from export.to_excel import summary_to_excel_classifier, summary_to_excel_regression
from config.settings import (
  BASE_DIR, EVALUATION_EXPORT_PATH,EXPORT_PATH,MODEL_EXPORT_PATH_LASSO,
  MODEL_EXPORT_PATH_LINEAR,MODEL_EXPORT_PATH_LOGISTIC,MODEL_EXPORT_PATH_RIDGE,
  VISUALIZATION_LASSO, VISUALIZATION_LINEAR, VISUALIZATION_LOGISTIC, VISUALIZATION_METRICS,
  VISUALIZATION_RIDGE, DATA_PATH
)
from visualization.regression_vis import plot_regression_metrics
from visualization.classifier_vis import plot_confusion_matrix, plot_roc_curve


def menu() :
  print("======== PROPERTI AI ========\n")
  print("1. Load and processing data")
  print("2. Train model")
  print("3. Evaluation model")
  print("4. save model evaluation results excel")
  print("5. Visualization Eval")
  print("6. Save visualization to PNG")
  print("7. Exit")

def main() :
  data = None
  x_c_train = x_c_test = y_c_train = y_c_test = None
  x_r_train = x_r_test = y_r_train = y_r_test = None
  model_logistic = None
  models = {}
  eval_logistic = None
  results_regression = {}

  while True :
    print()
    menu()
    pilihan = input("Choose (1-7) : ")

    if pilihan == '1' :
      try :
        print("[INFO] Data is being processed...")
        data = load_data(DATA_PATH)
        if data is None or data.empty:
          print("[ERROR] Loaded data is empty!")
          continue
        x_classifier, y_classifier = clean_data_classifier(data)
        x_regression, y_regression = clean_data_regression(data)
        # ket --> c = classifer, r = regression
        x_c_train, x_c_test, y_c_train, y_c_test = preprocessing(x_classifier, y_classifier, stratify=True)
        x_r_train, x_r_test, y_r_train, y_r_test = preprocessing(x_regression, y_regression)
        print("\n[INFO] Data loaded successfully...")

      except Exception as e :
        print(f"[ERROR] Failed to load and preprocess data... {e}")

    
    elif pilihan == "2" :

      if data is None :
        print("[WARNING] Data has not been found. Please select the first menu")
        continue

      print("[INFO] Model is being trained, wait a few seconds...")
      model_logistic = train_classifier(x_c_train, y_c_train)
      models = train_regressors(x_r_train, y_r_train)
      print("[INFO] Model has been trained")

    elif pilihan == "3" :
      if not all([model_logistic, models]) :
        print("[WARNING] Model has not been found. Please select the second menu")
        continue
      try :

          for name, model in models.items():
            y_r_pred = model.predict(x_r_test)
            result = evaluate_model_regression(name, y_r_test, y_r_pred)
            results_regression[name] = result
            print("======== EVALUATION REGRESSION MODEL (LINEAR, LASSO, RIDGE) ========\n")
            print(f"\nHasil evaluasi untuk model {name}:")
            print(f"  - MSE : {result['MSE']:.2f}")
            print(f"  - MAE : {result['MAE']:.2f}")
            print(f"  - RMSE   : {result['RMSE']:.2f}")
            print(f"  - R2 SCORE : {result['R2 Score']:.2f}")

          y_c_pred = model_logistic.predict(x_c_test)
          y_c_prob = model_logistic.predict_proba(x_c_test)[:, 1]
          eval_logistic = evaluate_model_classifier(y_c_test, y_c_pred, y_c_prob)
          print("\n\n======== EVALUATION CLASSIFIER MODEL (LOGISTIC) ========\n")
          print(f"  - Accuracy : {eval_logistic['Accuracy']:.2f}")
          print(f"  - Precision: {eval_logistic['Precision']:.2f}")
          print(f"  - Recall   : {eval_logistic['Recall']:.2f}")
          print(f"  - F1 Score : {eval_logistic['F1 Score']:.2f}")
          print(f"  - Confusion Matrix :\n {eval_logistic['Confusion Matrix']}")
          print(f"  - Report (Dict) :\n {eval_logistic['Report (dict)']}")
          print(f"  - Report (Text) :\n {eval_logistic['Report (text)']}")

      except Exception as e :
            print(f"[ERROR] Failed to evaluate model: {e}")

    elif pilihan == "4" :
      if not all([eval_logistic, results_regression]) :
        print("[WARNING] The model has not been evaluated. Please select the third menu")
        continue

      os.makedirs(os.path.dirname(MODEL_EXPORT_PATH_LINEAR), exist_ok=True)
      joblib.dump(models['linear'], MODEL_EXPORT_PATH_LINEAR)
      joblib.dump(models['ridge'], MODEL_EXPORT_PATH_RIDGE)
      joblib.dump(models['lasso'], MODEL_EXPORT_PATH_LASSO)
      joblib.dump(model_logistic, MODEL_EXPORT_PATH_LOGISTIC)

      eval_logistic_export = eval_logistic.copy()
      eval_logistic_export["Confusion Matrix"] = (
      eval_logistic["Confusion Matrix"].tolist()
      if isinstance(eval_logistic["Confusion Matrix"], np.ndarray)
      else eval_logistic["Confusion Matrix"]
      )
      summary_to_excel_classifier(eval_logistic_export, EVALUATION_EXPORT_PATH)
      summary_to_excel_classifier(eval_logistic, EVALUATION_EXPORT_PATH)
      summary_to_excel_regression(results_regression['linear'], results_regression['lasso'], results_regression['ridge'], EVALUATION_EXPORT_PATH)
      
      print(f"[INFO] All models are evaluated and saved...")
      print(f"[INFO] Evaluation finished at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    elif pilihan == '5' :
      if not all([eval_logistic, results_regression]):
        print("[WARNING] The model has not been evaluated. Please select the third menu")
        continue

      try:
        # Visualisasi metrik regresi
        plot_regression_metrics(results_regression)

        plot_confusion_matrix(eval_logistic['Confusion Matrix'], model_name='Logistic Regression')

        plot_roc_curve(
          eval_logistic['FPR'],
          eval_logistic['TPR'],
          eval_logistic['AUC Score'],
          model_name='Logistic Regression'
        )

      except Exception as e:
        print(f"[ERROR] Failed to generate visualization: {e}")

    elif pilihan == '6':
     if not all([eval_logistic, results_regression]):
        print("[WARNING] The model has not been evaluated. Please select the third menu first.")
        continue

     try:
        print("\n[INFO] Saving plots to PNG...")

        # Save plots as PNG
        plot_regression_metrics(results_regression, save=True, filename='regression_metrics.png')
        plot_confusion_matrix(eval_logistic['Confusion Matrix'], model_name='Logistic Regression', save=True, filename='confusion_matrix_logistic.png')
        plot_roc_curve(
            eval_logistic['FPR'],
            eval_logistic['TPR'],
            eval_logistic['AUC Score'],
            model_name='Logistic Regression',
            save=True,
            filename='roc_curve_logistic.png'
        )

        print(f"[SUCCESS] All plots saved to '{BASE_DIR}/output' folder.\n")

     except Exception as e:
        print(f"[ERROR] Failed to save visualizations: {e}")

      
    elif pilihan == "7" :
      print("THANKS!!!")
      break

if __name__ == "__main__" :
  main()

    