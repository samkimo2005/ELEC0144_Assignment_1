"""MLP experiments: A1 full 16x16, A6 full 8x8, A10 motion-only 16x16.

Runs ONLY the MLP (no CNN or GoogLeNet). Uses Training for fitting,
Validation for model comparison; add --test for held-out Test evaluation.
"""
import argparse
from pathlib import Path
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from Task_3_common import load_split, feature_extraction, evaluate_predictions, save_results


def make_features(split, resolution, mode):
    vectors = []
    print(f"Extracting {mode} features at {resolution}x{resolution}...", flush=True)
    for frames in split.frames:
        vector = feature_extraction(frames, resolution)
        if mode == "motion":
            vector = vector[-resolution * resolution:]
        vectors.append(vector)
    return np.asarray(vectors, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=int, choices=[8, 16], default=16)
    parser.add_argument("--features", choices=["all", "motion"], default="all")
    parser.add_argument("--test", action="store_true", help="Also evaluate held-out Test videos")
    parser.add_argument("--verbose", action="store_true", help="Show MLP training iterations")
    parser.add_argument("--no-cache", action="store_true", help="Read videos even if cached")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    train = load_split(root, "Training", use_cache=not args.no_cache)
    validation = load_split(root, "Validation", use_cache=not args.no_cache)
    train_features = make_features(train, args.resolution, args.features)
    validation_features = make_features(validation, args.resolution, args.features)

    # Identical MLP configuration to the original Task_3.py.
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    validation_features = scaler.transform(validation_features)
    mlp = MLPClassifier(hidden_layer_sizes=(256,), activation="relu",
                        max_iter=1000, random_state=42, verbose=args.verbose)
    print(f"Training MLP on {len(train.labels)} grips, {train_features.shape[1]} features", flush=True)
    mlp.fit(train_features, train.labels)
    results = {"Validation": evaluate_predictions(validation,
                   mlp.predict(validation_features), "Validation")}

    if args.test:
        test = load_split(root, "Test", use_cache=not args.no_cache)
        test_features = scaler.transform(make_features(test, args.resolution, args.features))
        results["Test"] = evaluate_predictions(test, mlp.predict(test_features), "Test")

    name = f"MLP_{args.resolution}x{args.resolution}_{args.features}"
    save_results(root, name, results)


if __name__ == "__main__":
    main()
