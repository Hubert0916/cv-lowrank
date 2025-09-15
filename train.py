# train.py
"""
Training and evaluation utilities for TinyCNN experiments.
Includes:
    - get_cifar10: Data loading pipeline with standard augmentation.
    - train_one_epoch: Single epoch training loop.
    - evaluate: Validation/testing loop (no gradient computation).
"""

import torch
from torchvision import datasets, transforms


def get_cifar10(batch_size: int = 128, num_workers: int = 4):
    """
    Prepare CIFAR-10 dataset loaders with standard augmentation.

    Args:
        batch_size (int): Batch size for training data.
        num_workers (int): Number of DataLoader worker processes.

    Returns:
        tuple: (train_loader, test_loader)
    """
    # Standard training augmentation: random crop + horizontal flip
    tf_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    # Test set: only convert to tensor (no augmentation)
    tf_test = transforms.Compose([transforms.ToTensor()])

    train_set = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=tf_train
    )
    test_set = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=tf_test
    )

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=256,      # use larger batch size for evaluation
        shuffle=False,
        num_workers=num_workers
    )
    return train_loader, test_loader


def train_one_epoch(model, loader, optimizer, device, criterion):
    """
    Train the model for one epoch.

    Args:
        model (nn.Module): The model to train.
        loader (DataLoader): Training data loader.
        optimizer (Optimizer): Optimizer instance (SGD, AdamW, etc.).
        device (str): Device to run on ("cuda" or "cpu").
        criterion: Loss function (e.g. CrossEntropyLoss).

    Returns:
        tuple: (average_loss, accuracy)
    """
    model.train()
    loss_sum, correct, n = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        # Forward pass
        out = model(x)
        loss = criterion(out, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate statistics
        loss_sum += loss.item() * x.size(0)
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        n += x.size(0)

    return loss_sum / n, correct / n


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    """
    Evaluate the model on a validation or test set.

    Args:
        model (nn.Module): The model to evaluate.
        loader (DataLoader): Validation/test data loader.
        device (str): Device to run on ("cuda" or "cpu").
        criterion: Loss function.

    Returns:
        tuple: (average_loss, accuracy)
    """
    model.eval()
    loss_sum, correct, n = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        # Forward pass only (no gradient)
        out = model(x)
        loss = criterion(out, y)

        # Accumulate statistics
        loss_sum += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        n += x.size(0)

    return loss_sum / n, correct / n
