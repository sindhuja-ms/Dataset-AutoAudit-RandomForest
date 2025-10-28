from src.data_loader import load_data
from src.train_model import train_model
from src.fairness_metrics import compute_class_bias
from src.visualize import plot_class_bias
import os

def main():
    print(" Loading dataset")
    X_train, X_test, y_train, y_test, le = load_data("data/weather_classification_data.csv")

    print(" Training model")
    model = train_model(X_train, y_train)

    print(" Evaluating class-wise bias")
    bias_df = compute_class_bias(model, X_test, y_test, le)
    print(bias_df)

    os.makedirs("outputs", exist_ok=True)
    print(" Generating bias plot and saving report")
    plot_class_bias(bias_df, "outputs/class_bias_plot.png")
    bias_df.to_csv("outputs/class_bias_report.csv", index=False)

    print(" Reports saved in outputs")

if __name__ == "__main__":
    main()
