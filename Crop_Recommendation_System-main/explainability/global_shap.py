# global_shap.py

import shap
import numpy as np
import matplotlib.pyplot as plt
from utils.label_formatting import format_feature

def generate_global_shap(model, X_background, feature_names, save_path):
    #generates GLOBAL SHAP feature-importance plot
    
    explainer = shap.TreeExplainer(model)

    X_sample = X_background.sample(
        n = min(200, len(X_background))
    )

    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        global_importance = np.mean(
            np.abs(shap_values),
            axis=(0, 2)
        )

    elif isinstance(shap_values, list):
        stacked = np.stack(
            [np.abs(v) for v in shap_values],
            axis=-1
        )
        global_importance = np.mean(stacked, axis=(0, 2))

    else:
        raise RuntimeError("Unexpected SHAP output formed")
    
    #sort features
    order = np.argsort(global_importance)[::-1]
    importance_sorted = global_importance[order]
    features_sorted = [format_feature(f) for f in np.array(feature_names)[order]]

    # Plot
    plt.figure(figsize=(8,4))
    y_pos = np.arange(len(features_sorted))

    plt.barh(y_pos, importance_sorted, color="#1565c0")
    plt.yticks(y_pos, features_sorted)
    
    plt.xlabel("Mean |SHAP Value|")
    plt.title("SHAP Global Feature Importance")
    plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()