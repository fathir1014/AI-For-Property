import pandas as pd
import os

def summary_to_excel_regression(linear,lasso,ridge,filepath) :
  os.makedirs(os.path.dirname(filepath), exist_ok=True)
  
  model_linear = pd.DataFrame([linear])
  model_ridge = pd.DataFrame([ridge])
  model_lasso = pd.DataFrame([lasso])

  with pd.ExcelWriter(filepath, engine='openpyxl') as writer :
    model_linear.to_excel(writer, sheet_name='Linear Regression', index=False)
    model_ridge.to_excel(writer, sheet_name='Ridge Regression', index=False)
    model_lasso.to_excel(writer, sheet_name='Lasso Regression', index=False)

def summary_to_excel_classifier(logistic, filepath) :

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    data = pd.DataFrame([logistic])
    data.to_excel(filepath, index=False)
    print(f"[INFO] Evaluation results are successfully saved in {filepath}...")

