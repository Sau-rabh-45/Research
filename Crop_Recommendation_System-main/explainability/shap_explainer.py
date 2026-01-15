# shap_explainer.py
import shap
import matplotlib.pyplot as plt
import numpy as np
from utils.label_formatting import format_feature, format_crop

def generate_shap_for_class(model, X_instance, class_index, class_name, feature_names, save_path, large = False):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_instance)

    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        values = shap_values[0, :, class_index]
    elif isinstance(shap_values, list):
        values = shap_values[class_index][0]
    else:
        raise RuntimeError(f"Unexpected SHAP output format")
    
    values = np.asarray(values, dtype=float)
    
    order = np.argsort(np.abs(values))[::-1]
    values_sorted = values[order]
    features_sorted = [format_feature(f) for f in np.array(feature_names)[order]]

    plt.figure(figsize=(10, 5) if large else (6, 3))
    y_pos = np.arange(len(features_sorted))
    colors = ["#2e7d32" if v>0 else "#c62828" for v in values_sorted]

    plt.barh(y_pos, values_sorted, color=colors)
    plt.yticks(y_pos, features_sorted)
    
    plt.xlabel("(Impact on prediction)")
    plt.title(f"SHAP Explanation for '{format_crop(class_name)}'")
    plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    explanation = []
    for f, v in zip(feature_names, values_sorted):
        if abs(v) > 0.01:
            direction = "increased" if v>0 else "decreased"
            explanation.append(
                f"{f} {direction} suitability ({v:.2f})"
            )
        
    return explanation