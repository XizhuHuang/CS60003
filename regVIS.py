import os
import matplotlib.pyplot as plt

def load_log(filepath):
    epochs = []
    train_acc = []
    val_acc = []

    with open(filepath, 'r') as f:
        lines = f.readlines()[1:]  # 跳过表头
        for line in lines:
            items = line.strip().split(',')
            epochs.append(int(items[0]))
            train_acc.append(float(items[2]))
            val_acc.append(float(items[4]))

    return epochs, train_acc, val_acc


def plot_overfitting():
    base_path = "logs"

    files = [
        ("reg=0.0", "lr0.1_h1024_512_reg0.0.txt"),
        ("reg=1e-4", "lr0.1_h1024_512_reg0.0001.txt"),
        ("reg=1e-3", "lr0.1_h1024_512_reg0.001.txt"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

    for i, (title, filename) in enumerate(files):
        filepath = os.path.join(base_path, filename)

        epochs, train_acc, val_acc = load_log(filepath)

        ax = axes[i]
        ax.plot(epochs, train_acc, label="Train Acc")
        ax.plot(epochs, val_acc, label="Val Acc")

        ax.set_title(title)
        ax.set_xlabel("Epoch")
        if i == 0:
            ax.set_ylabel("Accuracy")

        ax.grid(True)
        ax.legend()

    plt.suptitle("Overfitting Comparison under Different Regularization", fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)

    # 保存图片（论文可直接用）
    plt.savefig("outputs/overfitting_comparison.png", dpi=300)

    plt.show()


if __name__ == "__main__":
    plot_overfitting()