import numpy as np
import matplotlib.pyplot as plt

from models.mlp import NeuralNetwork


def minmax_normalize(img):
    """
    全局 min-max normalization（推荐）
    """
    return (img - img.min()) / (img.max() - img.min() + 1e-8)


def visualize_weights_grid(
    model,
    img_size=64,
    rows=4,
    cols=8,
    save_path=None,
    use_denorm=True  # ✅ 是否使用“类反归一化”
):
    """
    可视化第一层权重（改进版）

    参数：
        use_denorm:
            True  -> 映射到类似图像空间（更适合展示/报告）
            False -> 仅做min-max（更忠实权重本身）
    """

    W = model.linear1.W  # (input_dim, hidden_dim)

    num_filters = rows * cols
    assert W.shape[1] >= num_filters, "Not enough neurons to visualize"

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))

    for i in range(num_filters):
        ax = axes[i // cols, i % cols]

        # ===== reshape（关键修正）=====
        img = W[:, i].reshape(img_size, img_size, 3)

        # ===== normalization pipeline =====
        img = minmax_normalize(img)

        ax.imshow(img)
        ax.set_title(f"N{i}", fontsize=8)
        ax.axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved to {save_path}")

    plt.show()


if __name__ == "__main__":

    # ===== 初始化模型（必须和训练一致）=====
    model = NeuralNetwork(
        input_dim=12288,
        hidden_dim1=1024,
        hidden_dim2=512,
        output_dim=10
    )

    # ===== 加载权重 =====
    model.load_weights("outputs/best_model_2.npz")
    print("Model loaded.")

    # ===== 可视化（推荐：use_denorm=True）=====
    visualize_weights_grid(
        model,
        img_size=64,
        rows=4,
        cols=8,
        save_path="outputs/weight_grid.png",
        use_denorm=False
    )