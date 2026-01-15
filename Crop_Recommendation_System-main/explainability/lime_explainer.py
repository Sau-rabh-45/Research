# lime_explainer.py
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
from utils.label_formatting import format_feature, format_crop

def generate_lime_for_class(model, X_train, X_instance, feature_names, class_index, class_name, save_path, large=False):
    explainer = LimeTabularExplainer(
        training_data= X_train.values,
        feature_names=[format_feature(f) for f in feature_names],
        class_names=[format_crop(c) for c in model.classes_],
        mode="classification",
        random_state=42
    )

    exp = explainer.explain_instance(
        X_instance[0],
        model.predict_proba,
        labels=[class_index],
        num_features=len(feature_names)
    )

    fig = exp.as_pyplot_figure(label=class_index)
    if large:
        fig.set_size_inches(10, 5) 
    else:
        fig.set_size_inches(6, 3)

    plt.title(f"LIME - {format_crop(class_name)}")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()     