import os
import random
import matplotlib.pyplot as plt
from PIL import Image

def show_eurosat_samples(root_dir="data/EuroSAT_RGB", save_path="outputs/eurosat_samples.png"):
    
    class_names = [
        "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
        "Industrial", "Pasture", "PermanentCrop", "Residential",
        "River", "SeaLake"
    ]

    images = []
    labels = []

    # ===== 每个类别随机取几张 =====
    for cls in class_names:
        cls_path = os.path.join(root_dir, cls)
        img_list = [img for img in os.listdir(cls_path) if img.endswith(".jpg")]

        # 每类取2~3张，凑够24张
        selected = random.sample(img_list, 3)

        for img_name in selected:
            images.append(os.path.join(cls_path, img_name))
            labels.append(cls)

    # 只取前24张（4×6）
    images = images[:24]
    labels = labels[:24]

    # ===== 绘图 =====
    plt.figure(figsize=(12, 8))

    for i, (img_path, label) in enumerate(zip(images, labels)):
        plt.subplot(4, 6, i+1)

        img = Image.open(img_path)
        plt.imshow(img)
        plt.title(label, fontsize=8)
        plt.axis("off")

    plt.tight_layout()

    # ===== 保存图片 =====
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.show()

    print(f"Saved to {save_path}")


if __name__ == "__main__":
    show_eurosat_samples()