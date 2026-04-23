import numpy as np
import matplotlib.pyplot as plt

from models.mlp import NeuralNetwork


def generate_class_template_fast(
    model,
    class_idx,
    input_dim=12288,
    steps=300,
    lr=0.1,
    reg_lambda=0.01
):
    """
    使用反向传播计算 ∂x（高效）
    """

    # 初始化
    x = np.random.randn(1, input_dim) * 0.1

    for step in range(steps):

        # ===== forward =====
        logits = model.forward(x)
        z_c = logits[0, class_idx]

        # ===== 构造梯度 =====
        dlogits = np.zeros_like(logits)
        dlogits[0, class_idx] = 1.0   # ∂z_c / ∂logits

        # ===== 反传到输入 =====
        grad_x = model.backward_input(dlogits)  # (1, D)

        # ===== 加正则 =====
        grad_x = grad_x - 2 * reg_lambda * x

        # ===== 更新 =====
        x += lr * grad_x

        # ===== clamp =====
        x = np.clip(x, -1, 1)

        if (step + 1) % 50 == 0:
            print(f"Class {class_idx} | Step {step+1} | Logit {z_c:.4f}")

    return x[0]


def visualize_class_templates_fast(model, img_size=64, save_path=None):

    num_classes = 10
    templates = []

    print("Generating class templates (FAST)...")

    for c in range(num_classes):
        print(f"\n=== Class {c} ===")

        x = generate_class_template_fast(
            model,
            class_idx=c,
            steps=200,
            lr=0.2,
            reg_lambda=0.01
        )

        templates.append(x)

    # ===== 可视化 =====
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))

    for c in range(num_classes):
        ax = axes[c // 5, c % 5]

        img = templates[c].reshape(3, img_size, img_size).transpose(1, 2, 0)

        # normalize
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        ax.imshow(img)
        ax.set_title(f"Class {c}")
        ax.axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved to {save_path}")

    plt.show()


if __name__ == "__main__":

    model = NeuralNetwork(
        input_dim=12288,
        hidden_dim1=1024,
        hidden_dim2=512,
        output_dim=10
    )

    model.load_weights("outputs/best_model_2.npz")
    print("Model loaded.")

    visualize_class_templates_fast(
        model,
        img_size=64,
        save_path="outputs/class_templates_fast.png"
    )