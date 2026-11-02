"""Shared preprocessing, feature extraction, data loading, evaluation and CSV output.

Imported by MLP.py, CNN.py and GoogLeNet.py; does not train any model.
"""
from dataclasses import dataclass
from sklearn.metrics import accuracy_score, confusion_matrix
from pathlib import Path
import csv
import hashlib
import numpy as np
import cv2

VALID_EXTENSIONS = {".mp4", ".mov", ".avi"}
CACHE_VERSION = 1


@dataclass
class DataSplit:
    frames: list
    labels: np.ndarray
    objects: np.ndarray
    paths: list


def preprocessing(video_path):
    """Sample 12 frames from the middle 80% and return (12,224,224,3) RGB float32."""
    video = cv2.VideoCapture(str(video_path))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    if not video.isOpened() or total_frames == 0:
        video.release()
        raise ValueError("Cannot open the video: " + str(video_path))

    frame_numbers = np.linspace(0.1 * (total_frames - 1),
                                0.9 * (total_frames - 1), 12).astype(int)
    frames = []
    try:
        for number in frame_numbers:
            video.set(cv2.CAP_PROP_POS_FRAMES, int(number))
            success, frame = video.read()
            if not success:
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
    finally:
        video.release()
    return np.array(frames, dtype=np.float32)


def feature_extraction(frames, resolution=16):
    """Output three flattened maps: average edges, edge changes, mean motion."""
    edge_frames = []
    gray_frames = []
    for frame in frames:
        image = (frame * 255).astype(np.uint8)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        gray_frames.append(gray.astype(np.float32) / 255.0)
        edges = cv2.Canny(gray, 50, 150)
        edges = edges.astype(np.float32) / 255.0
        edges = cv2.resize(edges, (resolution, resolution),
                           interpolation=cv2.INTER_AREA)
        edge_frames.append(edges)

    edge_mean = np.mean(edge_frames, axis=0)
    first_edges = np.mean(edge_frames[:3], axis=0)
    last_edges = np.mean(edge_frames[-3:], axis=0)
    edge_change = np.abs(last_edges - first_edges)
    movements = []
    for i in range(1, len(gray_frames)):
        difference = np.abs(gray_frames[i] - gray_frames[i-1])
        difference = cv2.resize(difference, (resolution, resolution),
                                interpolation=cv2.INTER_AREA)
        movements.append(difference)
    movement_mean = np.mean(movements, axis=0)
    features = np.concatenate((edge_mean.flatten(),
                               edge_change.flatten(),
                               movement_mean.flatten()))
    return features.astype(np.float32)


def get_manifest(root):
    with open(root / "input_data" / "manifest.csv", "r", newline="",
              encoding="utf-8-sig") as file:
        rows = {row["clip_id"].strip().lower(): row
                for row in csv.DictReader(file)}
    return rows


def _cached_preprocessing(path, root):
    """Cache lossless 8-bit preprocessed frames; no video re-decode on subsequent runs."""
    info = path.stat()
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:12]
    cache_file = root / ".frame_cache" / f"{path.stem}_{digest}.npz"
    if cache_file.is_file():
        try:
            with np.load(cache_file, allow_pickle=False) as cached:
                if (int(cached["version"]) == CACHE_VERSION and
                    int(cached["mtime_ns"]) == info.st_mtime_ns and
                    int(cached["size"]) == info.st_size):
                    # Exact uint8 frames are sufficient: all preprocessed RGB values
                    # are integer pixel values divided by 255.
                    return cached["frames"].astype(np.float32) / 255.0
        except (OSError, ValueError, KeyError, EOFError):
            pass
    frames = preprocessing(path)
    cache_file.parent.mkdir(exist_ok=True, parents=True)
    np.savez_compressed(cache_file,
                        frames=np.rint(frames * 255).astype(np.uint8),
                        version=np.array(CACHE_VERSION),
                        mtime_ns=np.array(info.st_mtime_ns),
                        size=np.array(info.st_size))
    return frames


def load_split(root, split_name, use_cache=True):
    """Read only the requested split; every recording contributes exactly one label."""
    root = Path(root)
    folder = root / "input_data" / split_name
    if not folder.is_dir():
        raise FileNotFoundError(f"Dataset folder missing: {folder}")
    manifest = get_manifest(root)
    paths = sorted(p for p in folder.iterdir()
                   if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS)
    if not paths:
        raise ValueError(f"No videos found in {folder}")
    frames, labels, objects = [], [], []
    print(f"[{split_name}] {len(paths)} videos", flush=True)
    for index, path in enumerate(paths, 1):
        print(f"[{split_name}] {index}/{len(paths)}: {path.name}", flush=True)
        clip_id = path.stem.lower()
        if clip_id not in manifest:
            raise KeyError(f"Clip missing in manifest.csv: {clip_id}")
        record = manifest[clip_id]
        grip = record["grip"].strip().lower()
        if grip not in ("light", "firm"):
            raise ValueError("Unknown grip label: " + clip_id)
        frames.append(_cached_preprocessing(path, root) if use_cache
                      else preprocessing(path))
        labels.append(0 if grip == "light" else 1)
        objects.append(record["object"].strip().lower())
    return DataSplit(frames, np.asarray(labels, dtype=int),
                     np.asarray(objects), paths)


def select_cnn_frames(frame_lists):
    return np.concatenate([frames[[2, 5, 9]] for frames in frame_lists],
                          axis=0).astype(np.float32)


def evaluate_predictions(split, predictions, title):
    """Evaluate one prediction per video. 0=Light; 1=Firm."""
    predictions = np.asarray(predictions, dtype=int)
    if len(predictions) != len(split.labels):
        raise ValueError("Prediction count does not match the number of grip videos")
    accuracy = float(accuracy_score(split.labels, predictions))
    matrix = confusion_matrix(split.labels, predictions, labels=[0, 1])
    by_object = {}
    correct = int(np.sum(predictions == split.labels))
    print(f"\n[{title}] Accuracy: {accuracy:.2%} ({correct}/{len(split.labels)})", flush=True)
    print("Confusion matrix (rows=actual Light/Firm; columns=predicted Light/Firm):")
    print(matrix)
    for name in sorted(set(split.objects)):
        mask = split.objects == name
        obj_correct = int(np.sum(predictions[mask] == split.labels[mask]))
        value = float(accuracy_score(split.labels[mask], predictions[mask]))
        by_object[name] = value
        print(f"  {name}: {value:.2%} ({obj_correct}/{int(mask.sum())})")
    print("Predictions [Light, Firm]:", np.bincount(predictions, minlength=2))
    return {"accuracy": accuracy, "matrix": matrix,
            "by_object": by_object, "predictions": predictions}


def save_results(root, name, results):
    """Save one method's metrics without overwriting a different method's results."""
    output = Path(root) / "results"
    output.mkdir(parents=True, exist_ok=True)
    target = output / (name + ".csv")
    with target.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["Dataset", "Metric", "Value"])
        for split_name, result in results.items():
            writer.writerow([split_name, "Accuracy (%)", round(100 * result["accuracy"], 2)])
            for obj, value in result["by_object"].items():
                writer.writerow([split_name, obj + " Accuracy (%)", round(100 * value, 2)])
            for actual in range(2):
                for predicted in range(2):
                    writer.writerow([split_name,
                                     f"Actual {actual}, Predicted {predicted} (count)",
                                     int(result["matrix"][actual, predicted])])
    print("Results saved to:", target, flush=True)
    return target
