import matplotlib.pyplot as plt

def plot_class_bias(df, output_path="outputs/class_bias_plot.png"):
    plt.figure(figsize=(8, 4))
    plt.bar(df["class"], df["accuracy"], color="#4c72b0")
    plt.title("Per-Class Accuracy (Bias Indicator)")
    plt.xlabel("Class")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
