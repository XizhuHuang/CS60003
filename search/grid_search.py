import os
from models.mlp import NeuralNetwork
from train.trainer import Trainer

def grid_search(X_train, y_train, X_val, y_val):

    # ===== 创建日志目录 =====
    os.makedirs("logs", exist_ok=True)

    lrs = [0.1, 0.01]
    hidden_dims = [(256,128), (512,256), (1024,512)]
    regs = [0.0, 1e-4, 1e-3]

    best_acc = 0
    best_config = None

    for lr in lrs:
        for h1, h2 in hidden_dims:
            for reg in regs:

                print(f"\n==== Training: lr={lr}, h1={h1}, h2={h2}, reg={reg} ====")

                model = NeuralNetwork(
                    input_dim=X_train.shape[1],
                    hidden_dim1=h1,
                    hidden_dim2=h2,
                    output_dim=10
                )

                trainer = Trainer(
                    model,
                    lr=lr,
                    reg=reg,
                    step_size=10,   # 可以调小一点让grid search更快观察效果
                    gamma=0.5
                )

                history = trainer.train(
                    X_train, y_train,
                    X_val, y_val,
                    epochs=20,
                    batch_size=64,
                    verbose=False
                )

                # ===== 写日志 =====
                log_filename = f"logs/lr{lr}_h{h1}_{h2}_reg{reg}.txt"

                with open(log_filename, "w") as f:
                    f.write("epoch,train_loss,train_acc,val_loss,val_acc\n")

                    for i in range(len(history["train_losses"])):
                        f.write(
                            f"{i+1},"
                            f"{history['train_losses'][i]},"
                            f"{history['train_accs'][i]},"
                            f"{history['val_losses'][i]},"
                            f"{history['val_accs'][i]}\n"
                        )

                # ===== 更新最优参数 =====
                if trainer.best_val_acc > best_acc:
                    best_acc = trainer.best_val_acc
                    best_config = (lr, h1, h2, reg)

                print(f"[RESULT] Val Acc = {trainer.best_val_acc:.4f}")

    print("\n===== BEST RESULT =====")
    print("Best Config:", best_config)
    print("Best Val Acc:", best_acc)

    return best_config

if __name__ == "__main__":
    from data.eurosat import load_eurosat

    print("Loading EuroSAT dataset...")

    (X_train, y_train), (X_val, y_val), _ = load_eurosat("data/EuroSAT_RGB")

    best_config = grid_search(X_train, y_train, X_val, y_val)

    print("\nFinal Best Config:", best_config)