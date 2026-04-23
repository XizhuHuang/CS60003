import matplotlib.pyplot as plt
import os

def plot_all(trainer, save_path="outputs/curve.png", show=True):

    epochs = range(1, len(trainer.train_losses)+1)

    plt.figure(figsize=(10,8))

    # 1️⃣ Train Loss
    plt.subplot(2,2,1)
    plt.plot(epochs, trainer.train_losses)
    plt.title("Train Loss")

    # 2️⃣ Val Loss
    plt.subplot(2,2,2)
    plt.plot(epochs, trainer.val_losses)
    plt.title("Validation Loss")

    # 3️⃣ Train Acc
    plt.subplot(2,2,3)
    plt.plot(epochs, trainer.train_accs)
    plt.title("Train Accuracy")

    # 4️⃣ Val Acc
    plt.subplot(2,2,4)
    plt.plot(epochs, trainer.val_accs)
    plt.title("Validation Accuracy")

    plt.tight_layout()

    # ===== 保存图片 =====
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"[SAVE] Curve saved to {save_path}")

    # ===== 显示 =====
    if show:
        plt.show()
    else:
        plt.close()