import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils.label_formatting import format_feature, format_crop

def generate_cropwise_shap(model, X, feature_names, save_path):
    explainer = shap.TreeExplainer(model)

    X_sample = X.sample(n=min(300, len(X)), random_state=42)

    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, np.ndarray):
        n_classes = shap_values.shape[2]

        data = []
        for class_idx, crop in enumerate(model.classes_):
            mean_abs = np.mean(
                np.abs(shap_values[:, :, class_idx]),
                axis=0
            )
            data.append(mean_abs)

        df = pd.DataFrame(
            data,
            index=model.classes_,
            columns=feature_names
        )

        df.index = [format_crop(c) for c in df.index]
        df.columns = [format_feature(f) for f in df.columns]

    else:
        raise RuntimeError("Unexpected SHAP format")
    
    # Haetmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        df,
        cmap ="YlGnBu",
        annot=False
    )

    plt.title("Crop-wise Global Feature Impotance (SHAP)")
    plt.xlabel("Features")
    plt.ylabel("Crops")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    return df