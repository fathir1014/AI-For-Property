# 🏠 PropertiAI: Property Price Prediction & Credit Eligibility Classification

**PropertiAI** is a modular Machine Learning project designed to:
- ✅ Predict property prices based on key features (regression)
- ✅ Classify credit eligibility for house buyers (classification)

Built as a showcase portfolio project for real-world AI applications in the housing and finance industry.
---
## 📁 Project Structure

propertiAI/
├── config/ # Configuration and path settings
├── data/ # Input dataset (.xlsx)
├── loader/ # Data loading module
├── preprocessing/ # Data cleaning and preprocessing
├── model/ # ML model training and evaluation
├── predictor/ # Predict new data
├── visualization/ # Charts and metric visualizations
├── export/ # Save output files (Excel, PNG, etc.)
├── utils/ # Helper functions and logging
├── main.py # CLI interface to run the pipeline
└── README.md # This documentation
---
## 🔍 Features

### 1. **Property Price Prediction (Regression)**
- Models used: Linear Regression, Ridge, Lasso
- Metrics: MAE, MSE, R² Score
- Visualization: Comparison bar chart of evaluation metrics

### 2. **Credit Eligibility Classification**
- Model used: Logistic Regression
- Metrics: Confusion Matrix, ROC Curve, AUC Score
- Visualization: Confusion Matrix heatmap & ROC curve plot
---
## ⚙️ Installation

```bash
pip install -r requirements.txt

🛠️ How to Use
Run the project via CLI:

python main.py

You’ll see an interactive menu to:

- Preview dataset
- Train ML models
- Evaluate results
- Generate visualizations
- Save outputs to Excel or PNG

📊 Output Examples
All output files are saved in the output/ directory:

- regression_metrics.png – Regression metric comparison
- confusion_matrix_logistic.png – Classification confusion matrix
- roc_curve_logistic.png – ROC Curve for Logistic Regression
- evaluation_model.xlsx – Combined results saved to Excel

💡 Notes

- Dataset is stored in loan_data.xlsx
- Focused on clean modular code (no OOP yet)
- Fully extensible for future improvements (e.g., other classifiers, feature engineering)

🎯 Project Goals

This project serves as a professional portfolio example for:

- Freelancing on Upwork/Fiverr
- Demonstrating AI in real estate & finance sectors
- Applying real-world supervised learning

🙋‍♂️ Author
Developed by Fathir, aspiring AI expert with a mission to build practical, enterprise-grade intelligent systems.

📬 Contact
Want to collaborate or hire? Reach out via Upwork or GitHub.
Github : fathir1014
Upwork : Fathir Rizki Fadillah
Email : fathirrf80@gmail.com
