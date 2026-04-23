import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split


def load_eurosat(root_dir="data/EuroSAT_RGB", img_size=64):
    """
    从本地EuroSAT_RGB文件夹加载数据
    返回：
        (X_train, y_train), (X_val, y_val), (X_test, y_test)
    """

    X = []
    y = []

    # ===== 固定类别顺序（非常重要！避免label混乱）=====
    class_names = [
        "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
        "Industrial", "Pasture", "PermanentCrop", "Residential",
        "River", "SeaLake"
    ]

    class_to_idx = {cls: i for i, cls in enumerate(class_names)}

    for cls in class_names:
        cls_path = os.path.join(root_dir, cls)

        if not os.path.exists(cls_path):
            raise ValueError(f"路径不存在: {cls_path}")

        for img_name in os.listdir(cls_path):
            if not img_name.endswith(".jpg"):
                continue

            img_path = os.path.join(cls_path, img_name)

            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize((img_size, img_size))

                img = np.array(img).astype(np.float32) / 255.0
                img = (img - 0.5) / 0.5   # normalize到[-1,1]

                X.append(img.flatten())
                y.append(class_to_idx[cls])

            except Exception as e:
                print(f"读取失败: {img_path}, 错误: {e}")

    X = np.array(X)
    y = np.array(y)

    print(f"数据加载完成: {X.shape}, 类别数: {len(class_names)}")

    # ===== 数据划分 =====
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)