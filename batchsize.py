import numpy as np
import matplotlib.pyplot as plt
from models.mlp import NeuralNetwork
from train.trainer import Trainer
from data.eurosat import load_eurosat

# ===== 加载数据 =====
(X_train, y_train), (X_val, y_val), _ = load_eurosat("data/EuroSAT_RGB")


# ===== batch size 列表 =====
batch_sizes = [32, 64, 128, 256]

# ===== 颜色映射 =====
colors = {
    32: "blue",
    64: "green",
    128: "orange",
    256: "red"
}

# ===== 存储所有结果 =====
results = {}

# ===== 统一超参数 =====
EPOCHS = 50

for bs in batch_sizes:
    print(f"\n===== Training with batch_size = {bs} =====")

    # ===== 初始化模型（固定结构 256,128）=====
    model = NeuralNetwork(
        input_dim=X_train.shape[1],
        hidden_dim1=256,
        hidden_dim2=128,
        output_dim=10
    )

    # ===== Trainer（不关心保存路径）=====
    trainer = Trainer(
        model,
        lr=0.1,
        reg=1e-4,
        step_size=10,
        gamma=0.5,
        save_path="outputs/tmp.npz"   # 不会使用
    )

    # ===== 训练 =====
    history = trainer.train(
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=EPOCHS,
        batch_size=bs,
        verbose=False
    )

    # ===== 保存结果 =====
    results[bs] = history


# =====================================================
# 🎨 绘图（四个子图在一张图中）
# =====================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# ===== 1. Train Loss =====
for bs in batch_sizes:
    axes[0, 0].plot(
        results[bs]["train_losses"],
        label=f"bs={bs}",
        color=colors[bs]
    )
axes[0, 0].set_title("Train Loss")
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_ylabel("Loss")
axes[0, 0].legend()

# ===== 2. Train Accuracy =====
for bs in batch_sizes:
    axes[0, 1].plot(
        results[bs]["train_accs"],
        label=f"bs={bs}",
        color=colors[bs]
    )
axes[0, 1].set_title("Train Accuracy")
axes[0, 1].set_xlabel("Epoch")
axes[0, 1].set_ylabel("Accuracy")
axes[0, 1].legend()

# ===== 3. Validation Loss =====
for bs in batch_sizes:
    axes[1, 0].plot(
        results[bs]["val_losses"],
        label=f"bs={bs}",
        color=colors[bs]
    )
axes[1, 0].set_title("Validation Loss")
axes[1, 0].set_xlabel("Epoch")
axes[1, 0].set_ylabel("Loss")
axes[1, 0].legend()

# ===== 4. Validation Accuracy =====
for bs in batch_sizes:
    axes[1, 1].plot(
        results[bs]["val_accs"],
        label=f"bs={bs}",
        color=colors[bs]
    )
axes[1, 1].set_title("Validation Accuracy")
axes[1, 1].set_xlabel("Epoch")
axes[1, 1].set_ylabel("Accuracy")
axes[1, 1].legend()

plt.tight_layout()
plt.savefig("outputs/batchsize_comparison.png")
plt.show()