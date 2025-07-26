import os

# BASE FOLDER PROJECTS
BASE_DIR = "propertiAI"
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

# DATA OUTPUT MODEL
MODEL_EXPORT_PATH_LOGISTIC = os.path.join(BASE_DIR, 'model', 'logistic.pkl')
MODEL_EXPORT_PATH_RIDGE = os.path.join(BASE_DIR, 'model', 'ridge.pkl')
MODEL_EXPORT_PATH_LINEAR = os.path.join(BASE_DIR, 'model', 'linear.pkl')
MODEL_EXPORT_PATH_LASSO = os.path.join(BASE_DIR, 'model', 'lasso.pkl')
EVALUATION_EXPORT_PATH = os.path.join(BASE_DIR, 'output', 'evaluation_model.xlsx')

# DATA OUTPUT VISUALISASI
VISUALIZATION_LINEAR   = os.path.join(OUTPUT_DIR, 'viz_linear.png')
VISUALIZATION_RIDGE    = os.path.join(OUTPUT_DIR, 'viz_ridge.png')
VISUALIZATION_LASSO    = os.path.join(OUTPUT_DIR, 'viz_lasso.png')
VISUALIZATION_LOGISTIC = os.path.join(OUTPUT_DIR, 'confusion_matrix_logistic.png')
VISUALIZATION_METRICS  = os.path.join(OUTPUT_DIR, 'bar_chart_mse_mae_r2.png')

# GENERAL CONFIGS
TEST_SIZE = 0.2
RANDOM_STATE = 42
MAX_ITER = 1000
ALPHA = 1

DATA_PATH = os.path.join(BASE_DIR, 'data', 'data_properti_kredit.xlsx')
EXPORT_PATH = os.path.join(BASE_DIR, "export", "hasil_model.xlsx")