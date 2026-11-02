"""A10: Motion-only MLP (16x16 -> 256 features), with visible progress.

Place alongside input_data/ (Training/, Validation/, manifest.csv).
Only trains and validates A10; does not run CNNs, GoogLeNet, or Test.
Uses the same preprocessing, feature extraction, scaler and MLP settings
as the existing A1 implementation. The only new model setting is
verbose=True, which prints training progress without changing the algorithm.
"""

print("A10 started: importing libraries...", flush=True)

from pathlib import Path
from time import perf_counter
import csv

import cv2
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

print("Libraries loaded.", flush=True)


def preprocessing(video_path):
    """Same video preprocessing as the existing Task_3.py."""
    video = cv2.VideoCapture(str(video_path))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    if not video.isOpened() or total_frames == 0:
        video.release()
        raise ValueError("Cannot open the video: " + str(video_path))

    frame_numbers = np.linspace(
        0.1 * (total_frames - 1), 0.9 * (total_frames - 1), 12
    ).astype(int)
    frames = []

    for number in frame_numbers:
        video.set(cv2.CAP_PROP_POS_FRAMES, int(number))
        success, frame = video.read()
        if not success:
            video.release()
            raise ValueError("Cannot read a frame from: " + str(video_path))

        height, width = frame.shape[:2]
        scale = min(224 / width, 224 / height)
        new_width = max(1, int(width * scale))
        new_height = max(1, int(height * scale))
        frame = cv2.resize(frame, (new_width, new_height))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image = np.zeros((224, 224, 3), dtype=np.uint8)
        x = (224 - new_width) // 2
        y = (224 - new_height) // 2
        image[y:y + new_height, x:x + new_width] = frame
        frames.append(image / 255.0)

    video.release()
    return np.array(frames, dtype=np.float32)


def feature_extraction(frames):
    """Same 16x16 handcrafted features as A1; retain only motion later."""
    edge_frames = []
    gray_frames = []

    for frame in frames:
        image = (frame * 255).astype(np.uint8)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        gray_frames.append(gray.astype(np.float32) / 255.0)

        edges = cv2.Canny(gray, 50, 150)
        edges = edges.astype(np.float32) / 255.0
        edges = cv2.resize(edges, (16, 16), interpolation=cv2.INTER_AREA)
        edge_frames.append(edges)

    edge_mean = np.mean(edge_frames, axis=0)
    first_edges = np.mean(edge_frames[:3], axis=0)
    last_edges = np.mean(edge_frames[-3:], axis=0)
    edge_change = np.abs(last_edges - first_edges)

    movements = []
    for i in range(1, len(gray_frames)):
        difference = np.abs(gray_frames[i] - gray_frames[i - 1])
        difference = cv2.resize(
            difference, (16, 16), interpolation=cv2.INTER_AREA
        )
        movements.append(difference)
    movement_mean = np.mean(movements, axis=0)

    features = np.concatenate((
        edge_mean.flatten(),
        edge_change.flatten(),
        movement_mean.flatten(),
    ))
    return features.astype(np.float32)


def load_split(root, manifest, split):
    folder = root / "input_data" / split
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing video folder: {folder}")

    video_paths = sorted(
        p for p in folder.iterdir()
        if p.suffix.lower() in (".mp4", ".mov", ".avi")
    )
    total = len(video_paths)
    if not total:
        raise ValueError(f"No supported videos found in {folder}")

    print(f"\n[{split}] Starting {total} videos...", flush=True)
    feature_vectors = []
    labels = []

    for index, video_path in enumerate(video_paths, start=1):
        start = perf_counter()
        print(f"[{split}] {index}/{total}: {video_path.name} — processing...", flush=True)

        clip_id = video_path.stem.lower()
        if clip_id not in manifest:
            raise KeyError(f"Missing manifest entry: {clip_id}")
        grip = manifest[clip_id]["grip"].strip().lower()
        if grip not in ("light", "firm"):
            raise ValueError(f"Unknown grip label for {clip_id}: {grip}")

        frames = preprocessing(video_path)
        features = feature_extraction(frames)
        if features.shape != (768,):
            raise ValueError(
                f"Expected 768 features from 16x16 maps, got {features.shape} "
                f"for {clip_id}. Check feature_extraction()."
            )
        feature_vectors.append(features[-256:])  # Motion-only features
        labels.append(0 if grip == "light" else 1)

        print(
            f"[{split}] {index}/{total} complete "
            f"({100 * index / total:.0f}%, {perf_counter() - start:.1f}s)",
            flush=True,
        )

    print(f"[{split}] Finished: {total} grips.\n", flush=True)
    return np.asarray(feature_vectors, dtype=np.float32), np.asarray(labels)


def main():
    root = Path(__file__).resolve().parent
    manifest_path = root / "input_data" / "manifest.csv"
    print(f"Loading manifest: {manifest_path}", flush=True)
    with open(manifest_path, newline="", encoding="utf-8-sig") as file:
        manifest = {
            row["clip_id"].strip().lower(): row
            for row in csv.DictReader(file)
        }
    print(f"Manifest loaded: {len(manifest)} records.", flush=True)

    X_train, y_train = load_split(root, manifest, "Training")
    X_val, y_val = load_split(root, manifest, "Validation")

    print("Standardising features (fit on Training only)...", flush=True)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    print(f"Ready: {X_train.shape[1]} motion features per grip.", flush=True)

    mlp = MLPClassifier(
        hidden_layer_sizes=(256,),
        activation="relu",
        max_iter=1000,
        random_state=42,
        verbose=True,  # Display each MLP optimisation iteration; no change to training
    )
    print("\nTraining A10 motion-only MLP (iteration losses below)...", flush=True)
    mlp.fit(X_train, y_train)
    print(f"Training completed after {mlp.n_iter_} iterations.", flush=True)

    print("Evaluating Validation recordings...", flush=True)
    predictions = mlp.predict(X_val)

    print("\nA10: MLP with motion features only", flush=True)
    print(f"Features per grip: {X_train.shape[1]}")
    print(f"Training grips: {len(y_train)}; Validation grips: {len(y_val)}")
    print(f"Validation accuracy: {accuracy_score(y_val, predictions):.2%}")
    print("Validation confusion matrix (Light=0, Firm=1):")
    print(confusion_matrix(y_val, predictions, labels=[0, 1]))


if __name__ == "__main__":
    main()
