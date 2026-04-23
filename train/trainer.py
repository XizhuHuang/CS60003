import numpy as np
import os
from optim.sgd import SGDMomentum
from train.metrics import accuracy

class Trainer:
    def __init__(
        self,
        model,
        lr=0.01,
        reg=0.0,
        patience=10,
        step_size=20,   # 学习率衰减间隔
        gamma=0.5,      # 衰减系数
        seed=42,
        save_path = "outputs/best_model.npz"
    ):
        np.random.seed(seed)

        self.model = model
        self.lr = lr
        self.reg = reg
        self.patience = patience
        self.step_size = step_size
        self.gamma = gamma

        self.optimizer = SGDMomentum(model.layers, lr=lr, weight_decay=reg)

        # ===== early stopping =====
        self.best_val_acc = 0
        self.counter = 0
        self.save_path = save_path

        # ===== 训练记录（用于画图）=====
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []

        # ===== 创建输出目录 =====
        os.makedirs("outputs", exist_ok=True)

    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=64, verbose=True):

        for epoch in range(epochs):

            # ===== Learning Rate Decay (Step Decay) =====
            if (epoch + 1) % self.step_size == 0:
                self.optimizer.lr *= self.gamma
                if verbose:
                    print(f"[LR Decay] New LR = {self.optimizer.lr:.6f}")

            indices = np.random.permutation(len(X_train))

            train_loss_sum = 0.0
            train_correct = 0

            # ===== 训练 =====
            for i in range(0, len(X_train), batch_size):
                batch_idx = indices[i:i+batch_size]
                X_batch = X_train[batch_idx]
                y_batch = y_train[batch_idx]

                logits = self.model.forward(X_batch)

                loss = self.model.loss.forward(logits, y_batch)
                loss += self.model.get_regularization_loss(self.reg)

                self.model.backward()
                self.optimizer.step()

                # ===== 统计训练指标 =====
                train_loss_sum += loss * len(X_batch)
                train_correct += np.sum(np.argmax(logits, axis=1) == y_batch)

            # ===== 训练集指标 =====
            train_loss = train_loss_sum / len(X_train)
            train_acc = train_correct / len(X_train)

            # ===== 验证 =====
            val_logits = self.model.forward(X_val)
            val_loss = self.model.loss.forward(val_logits, y_val)
            val_loss += self.model.get_regularization_loss(self.reg)
            val_acc = accuracy(val_logits, y_val)

            # ===== 记录 =====
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)

            if verbose:
                print(f"Epoch {epoch+1}/{epochs} | "
                      f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

            # ===== 保存最优模型 =====
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.counter = 0

                self.model.save_weights(self.save_path)

            else:
                self.counter += 1
                if self.counter >= self.patience:
                    print("Early stopping triggered")
                    break

        # ===== 返回history（用于grid search日志）=====
        return {
            "train_losses": self.train_losses,
            "train_accs": self.train_accs,
            "val_losses": self.val_losses,
            "val_accs": self.val_accs
        }