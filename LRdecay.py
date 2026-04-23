import numpy as np
import matplotlib.pyplot as plt

from models.mlp import NeuralNetwork
from train.trainer import Trainer
from data.eurosat import load_eurosat


# ===== 加载数据 =====
(X_train, y_train), (X_val, y_val), _ = load_eurosat("data/EuroSAT_RGB")

EPOCHS = 50
BATCH_SIZE = 64

# =====================================================
# 🔧 三种LR调度函数
# =====================================================
def step_lr(epoch, base_lr, step_size=15, gamma=0.5):
    return base_lr * (gamma ** (epoch // step_size))

def cosine_lr(epoch, base_lr, total_epochs):
    return base_lr * 0.5 * (1 + np.cos(np.pi * epoch / total_epochs))

def linear_lr(epoch, base_lr, total_epochs, gamma=0.1):
    end_lr = base_lr * gamma
    return base_lr - (base_lr - end_lr) * (epoch / total_epochs)


# =====================================================
# 🧠 训练函数（支持自定义LR调度）
# =====================================================
def train_with_scheduler(scheduler_name):
    model = NeuralNetwork(
        input_dim=X_train.shape[1],
        hidden_dim1=256,
        hidden_dim2=128,
        output_dim=10
    )

    trainer = Trainer(
        model,
        lr=0.1,
        reg=1e-4,
        step_size=1000,  # 禁用内部StepLR
        gamma=1.0,
        save_path="outputs/tmp.npz"
    )

    history = {
        "train_losses": [],
        "val_losses": [],
        "val_accs": [],
        "lrs": []
    }

    base_lr = trainer.optimizer.lr

    for epoch in range(EPOCHS):

        # ===== 手动设置LR =====
        if scheduler_name == "step":
            lr = step_lr(epoch, base_lr)
        elif scheduler_name == "cosine":
            lr = cosine_lr(epoch, base_lr, EPOCHS)
        elif scheduler_name == "linear":
            lr = linear_lr(epoch, base_lr, EPOCHS)
        else:
            raise ValueError

        trainer.optimizer.lr = lr
        history["lrs"].append(lr)

        # ===== shuffle =====
        indices = np.random.permutation(len(X_train))

        train_loss_sum = 0
        train_correct = 0

        # ===== mini-batch训练 =====
        for i in range(0, len(X_train), BATCH_SIZE):
            batch_idx = indices[i:i+BATCH_SIZE]
            X_batch = X_train[batch_idx]
            y_batch = y_train[batch_idx]

            logits = trainer.model.forward(X_batch)

            loss = trainer.model.loss.forward(logits, y_batch)
            loss += trainer.model.get_regularization_loss(trainer.reg)

            trainer.model.backward()
            trainer.optimizer.step()

            train_loss_sum += loss * len(X_batch)
            train_correct += np.sum(np.argmax(logits, axis=1) == y_batch)

        train_loss = train_loss_sum / len(X_train)
        train_acc = train_correct / len(X_train)

        # ===== validation =====
        val_logits = trainer.model.forward(X_val)
        val_loss = trainer.model.loss.forward(val_logits, y_val)
        val_loss += trainer.model.get_regularization_loss(trainer.reg)

        val_acc = np.mean(np.argmax(val_logits, axis=1) == y_val)

        # ===== 记录 =====
        history["train_losses"].append(train_loss)
        history["val_losses"].append(val_loss)
        history["val_accs"].append(val_acc)

        print(f"[{scheduler_name}] Epoch {epoch+1} | "
              f"LR {lr:.4f} | Train Loss {train_loss:.4f} | Val Acc {val_acc:.4f}")

    return history


# =====================================================
# 🚀 运行三种策略
# =====================================================
results = {
    "StepLR": train_with_scheduler("step"),
    "CosineLR": train_with_scheduler("cosine"),
    "LinearLR": train_with_scheduler("linear")
}


# =====================================================
# 🎨 绘图（3 × 4）
# =====================================================
fig, axes = plt.subplots(3, 4, figsize=(16, 10))

for row, (name, hist) in enumerate(results.items()):

    # ===== Train Loss =====
    axes[row, 0].plot(hist["train_losses"])
    axes[row, 0].set_title(f"{name} - Train Loss")

    # ===== Val Loss =====
    axes[row, 1].plot(hist["val_losses"])
    axes[row, 1].set_title(f"{name} - Val Loss")

    # ===== Val Acc =====
    axes[row, 2].plot(hist["val_accs"])
    axes[row, 2].set_title(f"{name} - Val Accuracy")

    # ===== LR Curve =====
    axes[row, 3].plot(hist["lrs"])
    axes[row, 3].set_title(f"{name} - Learning Rate")

    for col in range(4):
        axes[row, col].set_xlabel("Epoch")

axes[0, 0].set_ylabel("StepLR")
axes[1, 0].set_ylabel("CosineLR")
axes[2, 0].set_ylabel("LinearLR")

plt.tight_layout()
plt.savefig("outputs/lr_scheduler_comparison.png")
plt.show()