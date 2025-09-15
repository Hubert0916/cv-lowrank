# main.py
"""
Main script to train and evaluate TinyCNN variants (Base, Low-Rank, Depthwise).
Results are collected and plotted as Accuracy vs. Model Size.
"""

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from models import TinyCNN
from train import get_cifar10, train_one_epoch, evaluate


def set_seed(seed: int = 42) -> None:
    """
    Fix random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_experiment(
    block: str = "base",
    rank: int = 16,
    epochs: int = 60,
    lr: float = 0.1,
    device: str = "cuda"
) -> dict:
    """
    Train a TinyCNN variant and return model statistics.

    Args:
        block (str): Type of convolution block ("base", "lowrank", "dwsep").
        rank (int): Rank for low-rank convolution (ignored for other blocks).
        epochs (int): Number of training epochs.
        lr (float): Learning rate.
        device (str): Device to train on ("cuda" or "cpu").

    Returns:
        dict: A dictionary with model name, rank, parameter count, and accuracy.
    """
    set_seed(0)
    train_loader, test_loader = get_cifar10()

    # Initialize model
    model = TinyCNN(stem=64, num_classes=10,
                    block=block, rank=rank).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model: {block} (rank={rank})  Params={params:,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=5e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    last_te_acc = 0.0
    for ep in range(1, epochs + 1):
        # Train one epoch
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, device, criterion
        )
        # Evaluate on validation set
        te_loss, te_acc = evaluate(model, test_loader, device, criterion)
        last_te_acc = te_acc * 100  # convert to percentage

        # Step scheduler
        scheduler.step()

        # Log every 10 epochs
        if ep % 10 == 0 or ep == 1:
            print(
                f"[{ep:03d}] train {tr_acc:.4f}/{tr_loss:.3f} | "
                f"test {te_acc:.4f}/{te_loss:.3f}"
            )

    return {
        "model": block,
        "rank": rank,
        "params": params,
        "acc": last_te_acc
    }


def plot_results(results: list) -> None:
    """
    Plot Accuracy vs. Parameters curve and save results to CSV.

    Args:
        results (list): A list of dictionaries with model stats.
    """
    df = pd.DataFrame(results)
    df["params_k"] = df["params"] / 1000

    plt.figure(figsize=(7, 4))
    colors = {"base": "#1f77b4", "lowrank": "#ff7f0e", "dwsep": "#2ca02c"}
    markers = {"base": "o", "lowrank": "s", "dwsep": "^"}

    for _, row in df.iterrows():
        label = row["model"].capitalize()
        if row["model"] == "lowrank":
            label += f" (r={row['rank']})"

        plt.scatter(
            row["params_k"],
            row["acc"],
            s=120,
            label=label,
            color=colors[row["model"]],
            marker=markers[row["model"]],
            edgecolors="black",
            linewidths=0.6
        )

    df_sorted = df.sort_values("params")
    plt.plot(df_sorted["params_k"], df_sorted["acc"],
             "--", color="gray", alpha=0.5)

    plt.title("CIFAR-10: Accuracy vs Model Size", fontsize=14)
    plt.xlabel("Parameters (K)", fontsize=12)
    plt.ylabel("Test Accuracy (%)", fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig("results_plot.png", dpi=300)
    print("Done. Saved plot to results_plot.png")
    plt.show()

    df.to_csv("results.csv", index=False)
    print("Done. Saved results to results.csv")


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    results = []

    # Base model
    results.append(run_experiment(block="base", epochs=60, device=dev))

    # Sweep multiple ranks to find sweet spot
    rank_sweep = [8, 16, 24, 32, 48, 64]
    for r in rank_sweep:
        results.append(run_experiment(block="lowrank",
                                      rank=r, epochs=60, device=dev))

    # Depthwise separable convolution
    results.append(run_experiment(block="dwsep", epochs=60, device=dev))

    plot_results(results)
