from data.eurosat import load_eurosat
from models.mlp import NeuralNetwork
from train.trainer import Trainer
from utils.plot import plot_all

import os

# ===== 创建目录 =====
os.makedirs("outputs", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# ===== 加载数据 =====
(X_train, y_train), (X_val, y_val), _ = load_eurosat("data/EuroSAT_RGB")

# ===== 初始化模型 =====
model = NeuralNetwork(
    input_dim=X_train.shape[1],
    hidden_dim1=1024,
    hidden_dim2=512,
    output_dim=10
)

# ===== 初始化 Trainer =====
trainer = Trainer(
    model,
    lr=0.1,
    reg=1e-4,
    step_size=20,   # 学习率衰减
    gamma=0.5,
    save_path="outputs/best_model.npz"
)

# ===== 训练 =====
history = trainer.train(
    X_train,
    y_train,
    X_val,
    y_val,
    epochs=50,
    batch_size=64,
    verbose=True
)

# =====================================================
# ✅ 1️⃣ 保存训练日志（模拟grid search的log格式）
# =====================================================
log_path = "logs/lr0.1_h1024_512_reg0.0001_epoch50.txt"

with open(log_path, "w") as f:
    f.write("epoch,train_loss,train_acc,val_loss,val_acc\n")

    for i in range(len(history["train_losses"])):
        f.write(
            f"{i+1},"
            f"{history['train_losses'][i]},"
            f"{history['train_accs'][i]},"
            f"{history['val_losses'][i]},"
            f"{history['val_accs'][i]}\n"
        )

print(f"\n训练日志已保存到: {log_path}")

# =====================================================
# ✅ 2️⃣ 绘制四张曲线（检查plot是否正确）
# =====================================================
plot_all(trainer)

# =====================================================
# ✅ 3️⃣ 打印最佳结果
# =====================================================
print("\n===== Training Finished =====")
print(f"Best Validation Accuracy: {trainer.best_val_acc:.4f}")