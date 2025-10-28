import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

def compute_class_bias(model, X_test, y_test, label_encoder):
    y_pred = model.predict(X_test)
    class_names = label_encoder.classes_

    cm = confusion_matrix(y_test, y_pred)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)

    df = pd.DataFrame({
        "class": class_names,
        "accuracy": per_class_acc
    })

    df["bias_indicator"] = df["accuracy"].max() - df["accuracy"]

    overall_acc = np.mean(y_pred == y_test)
    print(f"Overall model accuracy: {overall_acc:.3f}")
    return df
