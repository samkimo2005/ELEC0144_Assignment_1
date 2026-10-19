# %% Settings
"""Task 1 darkening experiment; add your existing CODE MAP above this line.

Run only AFTER the team has recorded its predictions. This file imports the
existing Task 1 functions and does not change their colour or growth settings.
"""

import csv
import importlib.util
import io
import warnings
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Save figures without opening 19 interactive windows.
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
SOURCE_NAME = "Task_1.py"
RESULTS_DIR = HERE / "WhatIf_Task1_Results"

# Keep this fixed to match the submitted Task 1 configuration.
MIRROR = False
DARK_FACTORS = [round(1.0 - 0.05 * i, 2) for i in range(19)]
FRUITS = ("Apple", "Capsicum", "Lemon")
COLOURS = {"Apple": "red", "Capsicum": "lime", "Lemon": "cyan"}


# %% Detection Experiment

def load_task1():
    """Load an existing Task 1 file from the same directory.

    Source file names are checked in SOURCE_NAMES order. Image paths remain
    exactly as defined by the original Task 1 file.
    """
    source = SOURCE_NAME
    if source is None:
        raise FileNotFoundError(
            f"Cannot find your Task 1 script in {HERE}. "
            "Place this file beside Task_1.py (or edit SOURCE_NAMES)."
        )

    spec = importlib.util.spec_from_file_location("task1_existing", source)
    task1 = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(task1)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"The Task 1 file '{source}' cannot load Fruits.jpg. "
            f"Missing path: {exc.filename}. Check the image_path in your "
            "original Task 1 script; keep the same working folder structure."
        ) from exc
    return task1


def detect_at_brightness(task1, dark_factor):
    """Run the original detection functions once, without modifying settings."""
    try:
        with redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")
            image = task1.preprocessing(dark_factor, MIRROR)
            apple, capsicum, lemon = task1.colour_segmentation(image)
            detections = {}
            for name, mask in zip(FRUITS, (apple, capsicum, lemon)):
                blob, features, candidates, seed = task1.process_mask(
                    mask, image, name, MIRROR
                )
                detections[name] = features
    finally:
        # colour_segmentation() creates a figure, even during batch runs.
        plt.close("all")
    return image, detections


# %% Result Recording

def make_row(dark_factor, name, features):
    """Record one fruit's numerical results without guessing visual accuracy."""
    row = {
        "darkFactor": f"{dark_factor:.2f}",
        "mirror": str(MIRROR),
        "fruit": name,
        "bbox": "",
        "bbox_area": "",
        "blob_area": "",
        "midpoint_x": "",
        "midpoint_y": "",
        "moment_x": "",
        "moment_y": "",
        "centroid_distance": "",
        "detection_review": "NO_BOX" if features is None else "REVIEW",
    }
    if features is None:
        return row

    xmin, ymin, xmax, ymax = (int(v) for v in features["bbox"])
    row.update({
        "bbox": f"({xmin}, {ymin}, {xmax}, {ymax})",
        "bbox_area": (xmax - xmin + 1) * (ymax - ymin + 1),
        "blob_area": int(features["area"]),
        "midpoint_x": f"{features['midpoint'][0]:.2f}",
        "midpoint_y": f"{features['midpoint'][1]:.2f}",
        "moment_x": f"{features['moment'][0]:.2f}",
        "moment_y": f"{features['moment'][1]:.2f}",
        "centroid_distance": f"{features['distance']:.2f}",
    })
    return row


# %% Detection Evaluation

def describe_review_requirement():
    """Explain the marking definition; do not confuse a box with a success."""
    print("\nIMPORTANT: REVIEW does not automatically mean FOUND.")
    print("A fruit is LOST if no box exists OR its box covers less than")
    print("half of the visible fruit. Inspect the contact sheets before")
    print("drawing conclusions about the first/last fruit to be lost.")


# %% Results Visualisation

def save_contact_sheets(previews):
    """Save images with all detected boxes for manual coverage inspection."""
    for start in range(0, len(previews), 5):
        batch = previews[start:start + 5]
        fig, axes = plt.subplots(len(batch), 1, figsize=(9, 3.8 * len(batch)))
        if len(batch) == 1:
            axes = [axes]

        for ax, (factor, image, detections) in zip(axes, batch):
            ax.imshow(image)
            for name, features in detections.items():
                if features is None:
                    continue
                xmin, ymin, xmax, ymax = features["bbox"]
                ax.add_patch(plt.Rectangle(
                    (xmin - .5, ymin - .5), xmax - xmin + 1, ymax - ymin + 1,
                    fill=False, edgecolor=COLOURS[name], linewidth=2,
                ))
                ax.text(xmin, max(8, ymin - 8), name, color="white", fontsize=9,
                        bbox={"facecolor": "black", "alpha": .65})
            ax.set_title(f"darkFactor = {factor:.2f} | mirror = {MIRROR}")
            ax.axis("off")

        fig.tight_layout()
        page = start // 5 + 1
        fig.savefig(RESULTS_DIR / f"contact_sheet_{page:02d}.png", dpi=150)
        plt.close(fig)


# %% Results Export

def save_csv(rows):
    """Export numerical measurements for subsequent What-if discussion."""
    with (RESULTS_DIR / "darkening_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# %% Main

def main():
    """Run the unchanged Task 1 pipeline at all 19 required brightness values."""
    task1 = load_task1()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    previews = []
    for factor in DARK_FACTORS:
        image, detections = detect_at_brightness(task1, factor)
        previews.append((factor, image, detections))
        for name in FRUITS:
            rows.append(make_row(factor, name, detections[name]))
        print(f"Processed darkFactor = {factor:.2f}")

    save_csv(rows)
    save_contact_sheets(previews)
    print(f"\nResults saved in: {RESULTS_DIR}")
    describe_review_requirement()


if __name__ == "__main__":
    main()
