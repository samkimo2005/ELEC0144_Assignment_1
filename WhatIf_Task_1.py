# CODE MAP of WhatIf_Task_1.py
# Settings              Define file locations, mirror mode, dark factors and plot colours.
# Detection Experiment  Import the submitted Task 1 pipeline and run it at each brightness.
# Result Recording      Convert each fruit's detected features into one CSV record.
# Detection Evaluation  Explain the manual Found/Lost criterion used in the report.
# Results Visualisation Save annotated contact sheets for checking fruit coverage.
# Results Export        Write the numerical results to darkening_results.csv.
# Main                  Run all 19 trials and save the results and contact sheets.
# AI use                GPT-6 (thinking setting not specified).

# %% Settings
"""Run the Task 1 What-if experiment with progressively darker images.

The experiment reuses the segmentation and region-growing functions from the
submitted Task_1.py rather than defining new detection thresholds. Each trial
applies a dark factor from 1.00 to 0.10, decreasing by 0.05, as specified in
the assignment. The mirror setting remains fixed for the entire experiment.

Run this experiment only after the team has recorded its predictions. The
output CSV contains numerical measurements, while the contact sheets support
manual assessment of whether each box covers at least half of the visible
fruit. The presence of a box alone does not establish successful detection.

Expected inputs:
    Task_1.py and the image files used by that script, in their existing paths.

Outputs (created in WhatIf_Task1_Results):
    darkening_results.csv and contact_sheet_01.png, etc.

Run from the directory containing Task_1.py and this script so that the
relative SOURCE_NAME path resolves correctly.
"""

import csv
import importlib.util
import io
import warnings
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Use a non-interactive backend during the 19-trial batch.
import matplotlib.pyplot as plt


# Locate outputs alongside the script, independently of the results folder's
# existence. SOURCE_NAME is a relative path, resolved from the current working
# directory when the source module is imported.
HERE = Path(__file__).resolve().parent
SOURCE_NAME = "Task_1.py"
RESULTS_DIR = HERE / "WhatIf_Task1_Results"

# Keep these settings unchanged across the 19 trials: changing them for
# individual factors would invalidate the fixed-parameters What-if comparison.
MIRROR = False
DARK_FACTORS = [round(1.0 - 0.05 * i, 2) for i in range(19)]
FRUITS = ("Apple", "Capsicum", "Lemon")
COLOURS = {"Apple": "red", "Capsicum": "lime", "Lemon": "cyan"}


# %% Detection Experiment

def load_task1():
    """Import the original Task 1 script as a module.

    The original script supplies ``preprocessing()``,
    ``colour_segmentation()`` and ``process_mask()``. Importing these functions
    lets the What-if script reuse their current thresholds, blob-selection
    rules and region-growing settings without duplicating them.

    Returns:
        module: Loaded Task 1 module containing the detection functions.

    Raises:
        FileNotFoundError: If importing Task 1 fails because a required image
            cannot be found. The message points to the original image path.

    Notes:
        ``SOURCE_NAME`` is resolved relative to the working directory.
        Run this script from the folder containing ``Task_1.py``.
    """
    source = SOURCE_NAME
    if source is None:
        raise FileNotFoundError(
            f"Cannot find your Task 1 script in {HERE}. "
            "Place this file beside Task_1.py (or edit SOURCE_NAMES)."
        )

    # Import by filename without calling the original script's main function.
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
    """Detect the three fruits after applying one darkening factor.

    Args:
        task1: Imported module implementing the submitted Task 1 functions.
        dark_factor (float): Multiplicative image-brightness factor for this
            trial, e.g. 0.30. Pixel rounding happens in Task 1 preprocessing.

    Returns:
        tuple:
            image (ndarray): Darkened, optionally mirrored RGB image.
            detections (dict): Maps each fruit name to its feature dictionary,
                or ``None`` if no candidate blob was selected.

    Notes:
        All masks and candidate blobs are computed by Task 1. Diagnostic
        console output is suppressed so the batch progress remains readable.
        Figures opened by Task 1's segmentation stage are closed after each
        trial, including when an exception occurs.
    """
    try:
        with redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")

            # Apply the requested darkening and the fixed mirror setting.
            image = task1.preprocessing(dark_factor, MIRROR)

            # Reuse the three original colour-segmentation masks.
            apple, capsicum, lemon = task1.colour_segmentation(image)
            detections = {}
            for name, mask in zip(FRUITS, (apple, capsicum, lemon)):
                # process_mask() applies the original component selection and
                # region growing; features describe the final detected blob.
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
    """Build one CSV record for one fruit at one dark factor.

    Args:
        dark_factor (float): Brightness factor for this trial.
        name (str): Fruit name: Apple, Capsicum or Lemon.
        features (dict | None): Final blob features returned by Task 1,
            including bounding box, two centroids, centroid distance and area.
            ``None`` indicates that no bounding box was produced.

    Returns:
        dict: One CSV row. Missing feature values are blank when no box was
        found, and the review field is then ``NO_BOX``. Otherwise the review
        field is ``REVIEW`` until fruit coverage has been judged visually.

    Notes:
        Bounding-box area includes both endpoint pixels:
        (xmax - xmin + 1) * (ymax - ymin + 1). This is *not* the same as the
        count of segmented pixels (``blob_area``), nor is it a measurement of
        visible-fruit coverage for the assignment's Found/Lost definition.
    """
    # Initialise consistent columns even for an undetected fruit. Avoid
    # automatically calling any existing box a successful detection.
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
    """Print the criterion required to determine Found versus Lost.

    A fruit is Lost if there is no box, or if the box covers less than half
    of its *visible fruit region*. Bounding-box size relative to another run
    is not equivalent to that coverage fraction. The code therefore retains
    ``REVIEW`` rather than making an unsupported automatic Found judgement.
    """
    print("\nIMPORTANT: REVIEW does not automatically mean FOUND.")
    print("A fruit is LOST if no box exists OR its box covers less than")
    print("half of the visible fruit. Inspect the contact sheets before")
    print("drawing conclusions about the first/last fruit to be lost.")


# %% Results Visualisation

def save_contact_sheets(previews):
    """Create contact sheets showing the boxes from each brightness trial.

    Args:
        previews (list): Sequence of ``(factor, image, detections)`` tuples,
            ordered by decreasing dark factor. ``detections`` maps fruit names
            to their final feature dictionaries, or ``None`` for no box.

    Side effects:
        Creates numbered PNG files in ``RESULTS_DIR``. Each page displays up
        to five trials with the same colour assignment for each fruit.

    Notes:
        Rectangle corners are offset by half a pixel and include the final
        pixel on each axis, matching the inclusive bounding-box area formula.
        These images are for manual review, not automatic coverage scoring.
    """
    # Separate the 19 trials into batches of at most five so labels and boxes
    # remain legible instead of crowding every trial into a single figure.
    for start in range(0, len(previews), 5):
        batch = previews[start:start + 5]
        fig, axes = plt.subplots(len(batch), 1, figsize=(9, 3.8 * len(batch)))
        if len(batch) == 1:
            axes = [axes]

        for ax, (factor, image, detections) in zip(axes, batch):
            ax.imshow(image)
            for name, features in detections.items():
                if features is None:
                    # With no detected blob there is no rectangle to draw.
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
    """Write all per-fruit measurements to ``darkening_results.csv``.

    Args:
        rows (list[dict]): Measurements prepared by ``make_row()``. Each row
            has the same set and order of keys, used as the CSV header.

    Side effects:
        Writes a UTF-8 CSV file in ``RESULTS_DIR``, replacing any older file
        with the same name. The expected full run contains 19 × 3 = 57 rows.
    """
    with (RESULTS_DIR / "darkening_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# %% Main

def main():
    """Run all 19 fixed-parameter darkening trials and export their outputs.

    Loads Task 1 once, processes each factor in descending order, and stores
    the image and resulting measurements for every fruit. Finally, writes the
    CSV, exports four contact sheets (5 + 5 + 5 + 4 trials), and prints the
    requirement for checking the Found/Lost status manually.

    This function does not alter Task 1's colour thresholds, blob-selection
    criteria or region-growing settings between trials.
    """
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
