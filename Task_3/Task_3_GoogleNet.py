"""Pretrained GoogLeNet visual features + logistic regression (original A5).

Runs only GoogLeNet. This requires torchvision and pretrained ImageNet weights.
Add --test to evaluate Test videos after Validation.
"""
import argparse
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.linear_model import LogisticRegression
from Task_3_common import load_split, select_cnn_frames, evaluate_predictions, save_results


def grip_features(model, images, number_of_grips):
    model.eval()
    device = next(model.parameters()).device
    features = []
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    with torch.no_grad():
        for i in range(0, len(images), 16):
            batch = torch.from_numpy(images[i:i+16]).permute(0, 3, 1, 2).to(device)
            features.append(model((batch - mean) / std).cpu().numpy())
    return np.concatenate(features, axis=0).reshape(number_of_grips, 3, -1).mean(axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", action="store_true", help="Also evaluate held-out Test videos")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    # Only GoogLeNet.py imports this pretrained architecture.
    from torchvision.models import googlenet as torchvision_googlenet, GoogLeNet_Weights
    root = Path(__file__).resolve().parent
    train = load_split(root, "Training", use_cache=not args.no_cache)
    validation = load_split(root, "Validation", use_cache=not args.no_cache)
    print("Loading pretrained GoogLeNet weights...", flush=True)
    googlenet = torchvision_googlenet(
        weights=GoogLeNet_Weights.DEFAULT, transform_input=False)
    googlenet.fc = nn.Identity()
    googlenet.eval()
    for parameter in googlenet.parameters():
        parameter.requires_grad = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    googlenet = googlenet.to(device)

    print("Extracting pretrained visual features...", flush=True)
    X_train = grip_features(googlenet, select_cnn_frames(train.frames), len(train.labels))
    X_val = grip_features(googlenet, select_cnn_frames(validation.frames), len(validation.labels))
    classifier = LogisticRegression(max_iter=1000, random_state=42)
    classifier.fit(X_train, train.labels)
    results = {"Validation": evaluate_predictions(validation,
                   classifier.predict(X_val), "Validation")}
    if args.test:
        test = load_split(root, "Test", use_cache=not args.no_cache)
        X_test = grip_features(googlenet, select_cnn_frames(test.frames), len(test.labels))
        results["Test"] = evaluate_predictions(test,
                   classifier.predict(X_test), "Test")
    save_results(root, "GoogLeNet", results)


if __name__ == "__main__":
    main()
