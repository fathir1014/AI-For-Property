import matplotlib.pyplot as plt
from config.settings import BASE_DIR
import os

def plot_regression_metrics(results_dict, save=False, filename='Regression_metrics.png'):
  
    metrics = ['MSE', 'MAE', 'RMSE', 'R2 Score']
    model_names = list(results_dict.keys())
    
    for metric in metrics:
        values = [results_dict[model][metric] for model in model_names]

        plt.figure(figsize=(8, 5))
        bars = plt.bar(model_names, values)
        plt.title(f'{metric} Comparison for Regression Models')
        plt.ylabel(metric)
        plt.xlabel('Model')
        plt.grid(axis='y', linestyle='--', alpha=0.6)

        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval, f"{yval:.2f}", ha='center', va='bottom')

        plt.tight_layout()

        if save:
          os.makedirs(f'{BASE_DIR}/output', exist_ok=True)
          plt.savefig(f"{BASE_DIR}/output/{filename}")
          print(f"[INFO] Saved: {BASE_DIR}output/{filename}")
        else:
           plt.show()
