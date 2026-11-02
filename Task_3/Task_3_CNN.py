"""Run one CNN experiment only: basic, BatchNorm or data augmentation.

Default: 5 convolutional layers / 40 epochs (latest Task_3.py settings).
For original A2: --method basic --layers 2 --epochs 10
For A8: --method basic --layers 5 --epochs 20
For A9: --method basic --layers 5 --epochs 40
"""
import argparse
from pathlib import Path
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from Task_3_common import load_split, select_cnn_frames, evaluate_predictions, save_results


def classification_cnn(method="basic", conv_layers=5):
    """Exactly the 2/5-convolution architecture of the supplied source."""
    channels = [3, 16, 32, 64, 64, 128]
    if conv_layers not in (2, 5):
        raise ValueError("Supported convolution counts: 2 or 5")
    layers = []
    for i in range(conv_layers):
        layers.append(nn.Conv2d(channels[i], channels[i + 1], 3, padding=1))
        if method == "batch":
            layers.append(nn.BatchNorm2d(channels[i + 1]))
        layers.append(nn.ReLU())
        if i in [0, 1, 3]:
            layers.append(nn.MaxPool2d(2))
    layers.extend([
        nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
        nn.Linear(channels[conv_layers], 32), nn.ReLU(),
        nn.Linear(32, 1), nn.Sigmoid()
    ])
    return nn.Sequential(*layers)


def grip_probabilities(model, images, num_grips):
    model.eval()
    device = next(model.parameters()).device
    probabilities = []
    with torch.no_grad():
        for i in range(0, len(images), 16):
            batch = torch.from_numpy(images[i:i+16]).permute(0, 3, 1, 2).to(device)
            probabilities.extend(model(batch).squeeze(1).cpu().numpy())
    return np.asarray(probabilities).reshape(num_grips, 3).mean(axis=1)


def grip_predictions(model, images, num_grips):
    return (grip_probabilities(model, images, num_grips) >= 0.5).astype(int)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["basic", "batch", "augmentation"], default="basic")
    parser.add_argument("--layers", type=int, choices=[2, 5], default=5)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=None,
                        help="Optional reproducibility seed; original runs had no fixed PyTorch seed")
    parser.add_argument("--test", action="store_true", help="Also evaluate held-out Test videos")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    root = Path(__file__).resolve().parent
    train = load_split(root, "Training", use_cache=not args.no_cache)
    validation = load_split(root, "Validation", use_cache=not args.no_cache)
    train_images = select_cnn_frames(train.frames)
    val_images = select_cnn_frames(validation.frames)
    train_labels = np.repeat(train.labels, 3)
    loader = DataLoader(TensorDataset(
        torch.from_numpy(train_images).permute(0, 3, 1, 2),
        torch.tensor(train_labels, dtype=torch.float32)
    ), batch_size=16, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn = classification_cnn(args.method, args.layers).to(device)
    augmentation = transforms.ColorJitter(brightness=0.1, contrast=0.1)
    optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)
    loss_function = nn.BCELoss()
    print(f"Training {args.method} CNN; {args.layers} layers; {args.epochs} epochs on {device}", flush=True)
    for epoch in range(args.epochs):
        cnn.train()
        epoch_loss = 0.0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if args.method == "augmentation":
                images = augmentation(images)
            optimizer.zero_grad()
            outputs = cnn(images).squeeze(1)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(labels)
        print(f"Epoch {epoch + 1}/{args.epochs} - loss: {epoch_loss/len(train_labels):.4f}", flush=True)

    train_prob = grip_probabilities(cnn, train_images, len(train.labels))
    val_prob = grip_probabilities(cnn, val_images, len(validation.labels))
    print("Training Firm probability (min, mean, max):", np.round(
        [train_prob.min(), train_prob.mean(), train_prob.max()], 4))
    print("Validation Firm probability (min, mean, max):", np.round(
        [val_prob.min(), val_prob.mean(), val_prob.max()], 4))
    print("Training accuracy:", f"{np.mean((train_prob>=0.5)==train.labels):.2%}")
    results = {"Validation": evaluate_predictions(validation,
               (val_prob >= 0.5).astype(int), "Validation")}

    if args.test:
        test = load_split(root, "Test", use_cache=not args.no_cache)
        test_images = select_cnn_frames(test.frames)
        results["Test"] = evaluate_predictions(test,
                   grip_predictions(cnn, test_images, len(test.labels)), "Test")
    save_results(root, f"CNN_{args.method}_{args.layers}layers_{args.epochs}epochs", results)


if __name__ == "__main__":
    main()
