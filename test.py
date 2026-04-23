import numpy as np
import matplotlib.pyplot as plt

from data.eurosat import load_eurosat
from models.mlp import NeuralNetwork
from train.metrics import accuracy, confusion_matrix


# ===== 类别名称（必须和dataset一致）=====
class_names = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
    "Industrial", "Pasture", "PermanentCrop", "Residential",
    "River", "SeaLake"
]


def softmax(logits):
    exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)


def show_errors(X, y, preds, indices, title="Error Cases", max_show=8):
    """
    可视化错误样本
    """
    num_show = min(len(indices), max_show)

    plt.figure(figsize=(12, 6))

    for i in range(num_show):
        idx = indices[i]

        img = X[idx].reshape(64, 64, 3)

        # ===== 反归一化 [-1,1] -> [0,1] =====
        img = (img * 0.5 + 0.5)
        img = np.clip(img, 0, 1)

        plt.subplot(2, 4, i + 1)
        plt.imshow(img)
        plt.title(f"GT: {class_names[y[idx]]}\nPred: {class_names[preds[idx]]}", fontsize=8)
        plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def show_confusion_examples(X, y, preds, confusion_pairs, class_names):
    """
    在一张图中展示多个典型混淆（每类1张）
    confusion_pairs: [(gt, pred), ...]
    """

    num = len(confusion_pairs)
    plt.figure(figsize=(3 * num, 3))

    count = 0

    for i, (gt, pred) in enumerate(confusion_pairs):

        # 找到该混淆对的样本
        mask = (y == gt) & (preds == pred)
        indices = np.where(mask)[0]

        if len(indices) == 0:
            continue  # 没有这种错误就跳过

        idx = indices[0]  # 只取1张

        img = X[idx].reshape(64, 64, 3)
        img = (img * 0.5 + 0.5)
        img = np.clip(img, 0, 1)

        plt.subplot(1, num, count + 1)
        plt.imshow(img)
        plt.title(f"GT: {class_names[gt]}\nPred: {class_names[pred]}", fontsize=8)
        plt.axis("off")

        count += 1

    plt.suptitle("Typical Confusion Cases")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    # ===== 加载数据 =====
    _, _, (X_test, y_test) = load_eurosat("data/EuroSAT_RGB")

    # ===== 初始化模型 =====
    model = NeuralNetwork(
        input_dim=X_test.shape[1],
        hidden_dim1=1024,
        hidden_dim2=512,
        output_dim=10
    )

    # ===== 加载权重 =====
    model.load_weights("outputs/best_model_2.npz")

    # ===== 推理 =====
    logits = model.forward(X_test)
    preds = np.argmax(logits, axis=1)

    # ===== 评估 =====
    acc = accuracy(logits, y_test)
    cm = confusion_matrix(logits, y_test, 10)

    print("Test Accuracy:", acc)
    print("Confusion Matrix:\n", cm)

    # =========================================================
    # 🔍 Error Analysis
    # =========================================================

    # ===== 找错误样本 =====
    wrong_idx = np.where(preds != y_test)[0]
    print(f"\nTotal wrong samples: {len(wrong_idx)}")

    # ===== 计算置信度 =====
    probs = softmax(logits)
    confidence = np.max(probs, axis=1)

    # =========================================================
    # 1️⃣ 高置信错误（最重要）
    # =========================================================
    wrong_conf = confidence[wrong_idx]
    top_wrong = wrong_idx[np.argsort(-wrong_conf)[:8]]

    show_errors(
        X_test, y_test, preds,
        top_wrong,
        title="Top Confident Wrong Predictions"
    )
    
    confusion_pairs = [
    (7, 4),  # Residential → Industrial
        (3, 8),  # Highway → River
        (2, 6),  # HerbaceousVegetation → PermanentCrop
        (3, 7),  # Highway → Residential
    ]

    show_confusion_examples(
        X_test, y_test, preds,
        confusion_pairs,
        class_names
    )