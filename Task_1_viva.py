# CODE MAP of Task_1.py
# 1. Imports and Configuration                             Import required libraries and define settings
# 2. Image Loading                                         Load Fruits.jpg
# 3. Preprocessing                                         Apply brightness scaling and horizontal mirroring
# 4. Colour Segmentation                                   Generate fruit masks using normalised RGB
# 5. Connected Component Analysis                          Find connected foreground components
# 6. Blob Selection                                        Filter and rank candidate fruit blobs
# 7. Region Growing                                        Refine fruit regions
# 8. Geometric Feature Extraction                          Compute bounding boxes, centroids and areas
# 9. Fruit Processing Pipeline                             Run detection and feature extraction per fruit
# 10. Final Results Visualisation                          Display and save final detection results
# 11. Main Pipeline                                        Execute the complete processing pipeline
# AI use                                                   ChatGPT GPT-6

# %% 1. Imports and Configuration

from pathlib import Path

from collections import deque

from matplotlib.lines import Line2D

import numpy as np

import matplotlib.pyplot as plt

# %% Variable Definition of Dark Factor and Mirror 

darkFactor = 0.6

mirror = True

# %% 2. Image Loading

folder = Path(__file__).resolve().parent

image_path = folder / "Assignment_Files_Draft37" / "Fruits.jpg"

fruit = plt.imread(image_path).astype(float)

# %% 3. Preprocessing

def preprocessing(darkFactor, mirror):

    """Apply brightness scaling and optional horizontal mirroring.

    Parameters
    ----------
    darkFactor : float
        Multiplicative brightness factor for the globally loaded ``fruit`` image.
        A value of 1.0 preserves brightness; lower values darken the image.
    mirror : bool
        Whether to reverse the image horizontally.

    Returns
    -------
    numpy.ndarray
        Rounded, clipped RGB image with dtype uint8 and shape (H, W, 3).

    Notes
    -----
    This function reads the module-level ``fruit`` variable. It does not
    modify that original array.
    """

    fruit_processed = np.clip(

        np.rint(fruit.astype(float) * darkFactor),

        0, 255

    ).astype(np.uint8)

    if mirror:

        fruit_processed = fruit_processed[:, ::-1, :].copy()

    return fruit_processed

# %% 4. Colour Segmentation

def colour_segmentation(image):

    """Create fruit-specific binary masks using normalised RGB thresholds.

    Each RGB channel is divided by the sum of all three channels, with a
    small denominator safeguard. Separate experimentally selected colour
    conditions identify candidate apple, capsicum, and lemon pixels.

    Parameters
    ----------
    image : numpy.ndarray
        Input RGB image with shape (H, W, 3).

    Returns
    -------
    apple_mask, capsicum_mask, lemon_mask : numpy.ndarray of bool
        Three two-dimensional masks of candidate fruit pixels.

    Notes
    -----
    Colour similarity alone does not guarantee correct object selection.
    """

    image_float = image.astype(float)

    R = image_float[:, :, 0]

    G = image_float[:, :, 1]

    B = image_float[:, :, 2]

    total = R + G + B

    r = R / np.maximum(total, 1e-8)

    g = G / np.maximum(total, 1e-8)

    b = B / np.maximum(total, 1e-8)

    # Initial threshold experiments

    apple_mask = (

        (r > 0.44) &

        (g > 0.08) & (g < 0.28) &

        (b > 0.14) & (b < 0.36) &

        (r > b + 0.14)

    )

    capsicum_mask = (

        (r > 0.50) &

        (g > 0.29) & (g < 0.42) &

        (b < 0.12) &

        (r > g * 1.45)

    )

    lemon_mask = (

        (r > 0.40) &

        (g > 0.36) &

        (b < 0.205) &

        (np.abs(r - g) < 0.11)

    )

    return apple_mask, capsicum_mask, lemon_mask

# %% 5. Connected Component Analysis

def connected_components(mask):

    """Find four-connected foreground regions in a binary mask.

    The implementation scans the image and uses an explicit stack for
    depth-first traversal. Only horizontal and vertical neighbours are
    considered; diagonal adjacency does not connect two pixels.

    Parameters
    ----------
    mask : numpy.ndarray of bool
        Two-dimensional binary image; True marks foreground pixels.

    Returns
    -------
    list of numpy.ndarray
        Connected components, each stored as an (N, 2) array of pixel
        coordinates in (row, column), equivalently (y, x), order.

    Notes
    -----
    All components are returned without filtering or ranking.
    """

    height, width = mask.shape

    visited = np.zeros_like(mask, dtype=bool)

    components = []

    for y in range(height):

        for x in range(width):

            if not mask[y, x] or visited[y, x]:

                continue

            stack = [(y, x)]

            visited[y, x] = True

            pixels = []

            while stack:

                cy, cx = stack.pop()

                pixels.append((cy, cx))

                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:

                    ny, nx = cy + dy, cx + dx

                    if (

                        0 <= ny < height and

                        0 <= nx < width and

                        mask[ny, nx] and

                        not visited[ny, nx]

                    ):

                        visited[ny, nx] = True

                        stack.append((ny, nx))

            components.append(np.array(pixels))

    return components

# %% 6. Blob Selection

def select_best_blob(components, image_shape):

    """Filter and rank connected components to select a candidate fruit.

    Components below 0.2% of the image area, touching an image border,
    or having an aspect ratio outside (0.5, 3.0) are discarded. Remaining
    components are ranked by ``area * extent``, where extent is the
    fraction of the component's bounding rectangle occupied by pixels.

    Parameters
    ----------
    components : list of numpy.ndarray
        Components with (y, x) pixel coordinates.
    image_shape : tuple
        Image shape; its first two elements are height and width.

    Returns
    -------
    best_blob : numpy.ndarray or None
        Highest-scoring component, or None if no component qualifies.
    candidates : list of tuple
        Descending list of (score, blob) pairs.

    Notes
    -----
    This is a geometric heuristic rather than semantic object recognition.
    """

    H, W = image_shape[:2]

    candidates = []

    for blob in components:

        ys = blob[:, 0]

        xs = blob[:, 1]

        area = len(blob)

        xmin, xmax = xs.min(), xs.max()

        ymin, ymax = ys.min(), ys.max()

        width = xmax - xmin + 1

        height = ymax - ymin + 1

        # Area filtering

        if area < H * W * 0.002:

            continue

        # Border filtering

        if (

            xmin == 0 or xmax == W - 1 or

            ymin == 0 or ymax == H - 1

        ):

            continue

        # Aspect ratio filtering

        aspect_ratio = width / height

        if not (0.5 < aspect_ratio < 3.0):

            continue

        # Scoring

        extent = area / (width * height)

        score = area * extent

        candidates.append((score, blob))

    # Sort by score, highest first

    candidates.sort(

        key=lambda item: item[0],

        reverse=True

    )

    # Select the best blob

    best_blob = (

        candidates[0][1] if candidates else None

    )

    return best_blob, candidates

# %% 7. Region Growing

def region_growing(

    image,

    seed_blob,

    fruit_name,

    margin_top=0.10,

    margin_bottom=0.10,

    margin_left=0.10,

    margin_right=0.10,

    max_area_factor=1.35,

    colour_distance_threshold=0.08,

    mirror=False

):

    """Expand a selected blob using constrained four-neighbour region growing.

    Starting from all seed pixels, a FIFO queue explores neighbouring pixels.
    A pixel is accepted only when it is within the search boundaries,
    satisfies the fruit-specific normalised RGB condition, and passes the
    optional Euclidean colour-distance test relative to the seed mean.
    Growth stops when the queue is empty or the area limit is reached.

    Parameters
    ----------
    image : numpy.ndarray
        Input RGB image with shape (H, W, 3).
    seed_blob : numpy.ndarray
        Initial component with (N, 2) coordinates in (y, x) order.
    fruit_name : str
        One of 'Apple', 'Capsicum', or 'Lemon'.
    margin_top, margin_bottom : float, optional
        Search extensions as fractions of the seed bounding-box height.
    margin_left, margin_right : float, optional
        Search extensions as fractions of the seed bounding-box width.
    max_area_factor : float, optional
        Upper bound on grown area relative to the seed pixel count.
    colour_distance_threshold : float or None, optional
        Maximum Euclidean distance in normalised RGB from the seed mean;
        None disables this additional similarity test.
    mirror : bool, optional
        Reverse the asymmetric horizontal search rules for mirrored images.

    Returns
    -------
    numpy.ndarray or None
        Final region coordinates in (y, x) order, or None for an empty seed.

    Notes
    -----
    The capsicum refinement limits newly accepted pixels to a lower region;
    the lemon refinement limits them to a direction-dependent side region.
    All original seed pixels remain included.
    Thresholds and spatial limits are empirically selected for this image.
    """

    if seed_blob is None or len(seed_blob) == 0:

        return None

    seed_blob = np.asarray(seed_blob, dtype=int)

    H, W = image.shape[:2]

    # Step 6.1: Normalised RGB

    pixels = image.astype(float)

    R = pixels[:, :, 0]

    G = pixels[:, :, 1]

    B = pixels[:, :, 2]

    total = np.maximum(R + G + B, 1e-8)

    r = R / total

    g = G / total

    b = B / total

    # Step 6.2: Original colour conditions

    if fruit_name == "Apple":

        allowed = (

            (r > 0.40) &

            (g > 0.07) & (g < 0.32) &

            (b > 0.10) & (b < 0.38) &

            (r > b + 0.08)

        )

    elif fruit_name == "Capsicum":

        allowed = (

            (r > 0.48) &

            (g > 0.25) & (g < 0.43) &

            (b < 0.14) &

            (r > 1.35 * g)

        )

    elif fruit_name == "Lemon":

        allowed = (

            (r > 0.34) &

            (g > 0.30) &

            (b < 0.27) &

            (np.abs(r - g) < 0.18)

        )

    else:

        raise ValueError(f"Unknown fruit: {fruit_name}")

    # Step 6.3: Seed properties

    ys = seed_blob[:, 0]

    xs = seed_blob[:, 1]

    mean_r = np.mean(r[ys, xs])

    mean_g = np.mean(g[ys, xs])

    mean_b = np.mean(b[ys, xs])

    xmin = int(xs.min())

    xmax = int(xs.max())

    ymin = int(ys.min())

    ymax = int(ys.max())

    width = xmax - xmin + 1

    height = ymax - ymin + 1

    # Step 6.4: Direction-constrained colour refinement

    Y, X = np.ogrid[:H, :W]

    if fruit_name == "Capsicum":

        # Restrict NEW growth to the lower part

        lower_region = (

            Y >= ymin + int(0.55 * height)

        )

        # Relax the colour criteria for orange pixels

        relaxed_colour = (

            (r > 0.43) &

            (g > 0.13) & (g < 0.45) &

            (b < 0.19) &

            (r > 1.20 * g)

        )

        allowed = (

            lower_region &

            (allowed | relaxed_colour)

        )

    elif fruit_name == "Lemon":

        # Right side in original image;

        # left side when mirrored.

        if mirror:

            side_region = (

                X <= xmax - int(0.55 * width)

            )

        else:

            side_region = (

                X >= xmin + int(0.55 * width)

            )

        # Relax colour criteria for orange-yellow peel

        relaxed_colour = (

            (r > 0.34) &

            (g > 0.24) &

            (b < 0.27) &

            (np.abs(r - g) < 0.28)

        )

        allowed = (

            side_region &

            (allowed | relaxed_colour)

        )

    # Step 6.5: Search boundaries

    if mirror:

        margin_left, margin_right = (

            margin_right, margin_left

        )

    left = max(

        0, xmin - int(width * margin_left)

    )

    right = min(

        W - 1, xmax + int(width * margin_right)

    )

    top = max(

        0, ymin - int(height * margin_top)

    )

    bottom = min(

        H - 1, ymax + int(height * margin_bottom)

    )

    # Step 6.6: Prepare region growing

    grown = np.zeros((H, W), dtype=bool)

    grown[ys, xs] = True

    queue = deque(

        (int(y), int(x)) for y, x in seed_blob

    )

    max_area = max(

        len(seed_blob),

        int(len(seed_blob) * max_area_factor)

    )

    area = len(seed_blob)

    # Colour similarity

    if colour_distance_threshold is not None:

        colour_distance = np.sqrt(

            (r - mean_r) ** 2 +

            (g - mean_g) ** 2 +

            (b - mean_b) ** 2

        )

    # Step 7: Region growing (4-connectivity)

    directions = [

        (-1, 0),

        (1, 0),

        (0, -1),

        (0, 1)

    ]

    while queue and area < max_area:

        y, x = queue.popleft()

        for dy, dx in directions:

            ny = y + dy

            nx = x + dx

            # Search boundary

            if not (

                top <= ny <= bottom and

                left <= nx <= right

            ):

                continue

            # Already accepted

            if grown[ny, nx]:

                continue

            # Colour condition

            if not allowed[ny, nx]:

                continue

            # Similarity condition

            if (

                colour_distance_threshold is not None

                and

                colour_distance[ny, nx] >

                colour_distance_threshold

            ):

                continue

            # Accept pixel

            grown[ny, nx] = True

            queue.append((ny, nx))

            area += 1

            if area >= max_area:

                break

    grown_blob = np.argwhere(grown)

    return grown_blob

# %% 8. Geometric Feature Extraction

def calculate_blob_features(blob):

    """Compute the bounding box, two centroids, and area of a blob.

    The bounding-box midpoint is the average of the extreme x and y
    coordinates. The moment centroid uses the binary spatial moments
    M00 = N, M10 = sum(x), and M01 = sum(y), giving (M10/M00, M01/M00).

    Parameters
    ----------
    blob : numpy.ndarray or None
        Pixel coordinates in (y, x) order, shape (N, 2).

    Returns
    -------
    dict or None
        Dictionary with 'bbox' (xmin, ymin, xmax, ymax), 'midpoint'
        (xb, yb), 'moment' (xm, ym), 'distance' (pixel distance between
        centroids), and 'area' (pixel count). Returns None for an empty blob.
    """

    if blob is None or len(blob) == 0:

        return None

    ys = blob[:, 0]

    xs = blob[:, 1]

    xmin, xmax = xs.min(), xs.max()

    ymin, ymax = ys.min(), ys.max()

    # Bounding box midpoint

    xb = (xmin + xmax) / 2

    yb = (ymin + ymax) / 2

    # Moment centroid

    M00 = len(xs)

    M10 = np.sum(xs)

    M01 = np.sum(ys)

    xm = M10 / M00

    ym = M01 / M00

    distance = np.sqrt((xb - xm) ** 2 + (yb - ym) ** 2)

    return {

        "bbox": (xmin, ymin, xmax, ymax),

        "midpoint": (xb, yb),

        "moment": (xm, ym),

        "distance": distance,

        "area": M00

    }

# %% 9. Fruit Processing Pipeline

def process_mask(mask, image, fruit_name, mirror=False):

    # Connected Components

    """Run component detection, selection, and refinement for one fruit.

    The binary mask is separated into four-connected components. The best
    candidate is selected using geometric scoring, then expanded with the
    existing fruit-specific region-growing parameters.

    Parameters
    ----------
    mask : numpy.ndarray of bool
        Fruit-specific candidate pixel mask.
    image : numpy.ndarray
        RGB image used for colour-constrained region growing.
    fruit_name : str
        One of 'Apple', 'Capsicum', or 'Lemon'.
    mirror : bool, optional
        Whether the image has been horizontally mirrored.

    Returns
    -------
    grown_blob : numpy.ndarray or None
        Final region coordinates in (y, x) order.
    Notes
    -----
    When no candidate qualifies, the function returns None.
    """

    components = connected_components(mask)

    # Blob Selection

    best_blob, _ = select_best_blob(

        components, mask.shape

    )

    if best_blob is None:

        print(f"Warning: {fruit_name} not detected.")

        return None

    # Fruit-specific Region Growing

    if fruit_name == "Apple":

        grown_blob = region_growing(

            image,

            best_blob,

            fruit_name,

            margin_top=0.12,

            margin_bottom=0.12,

            margin_left=0.12,

            margin_right=0.12,

            max_area_factor=1.25,

            colour_distance_threshold=0.07,

            mirror=mirror

        )

    elif fruit_name == "Capsicum":

        grown_blob = region_growing(

            image,

            best_blob,

            fruit_name,

            margin_top=0.02,

            margin_bottom=0.25,

            margin_left=0.05,

            margin_right=0.08,

            max_area_factor=1.90,

            colour_distance_threshold=None,

            mirror=mirror

        )

    elif fruit_name == "Lemon":

        grown_blob = region_growing(

            image,

            best_blob,

            fruit_name,

            margin_top=0.05,

            margin_bottom=0.08,

            margin_left=0.05,

            margin_right=0.25,

            max_area_factor=1.50,

            colour_distance_threshold=None,

            mirror=mirror

        )

    else:

        raise ValueError(f"Unknown fruit: {fruit_name}")

    return grown_blob

# %% 10. Final Results Visualisation

def visualise_results(

    image,

    apple_blob,

    capsicum_blob,

    lemon_blob,

    darkFactor,

    mirror,

    save_path="Task1_Final_Results.png",

):

    # Define fruit blobs and display colours

    """Display, save, and report the final detections for all three fruits.

    Each detected fruit is outlined with a labelled bounding box. A white
    cross marks the bounding-box midpoint and a coloured circle marks the
    moment centroid. The figure is saved at 300 DPI when a path is given.

    Parameters
    ----------
    image : numpy.ndarray
        RGB image used as the background.
    apple_blob, capsicum_blob, lemon_blob : numpy.ndarray or None
        Final fruit regions, with coordinates in (y, x) order.
    save_path : str, pathlib.Path, or None, optional
        Destination for the figure; None skips saving.

    Returns
    -------
    dict
        Mapping from fruit name to its geometric features, or None if the
        corresponding fruit was not detected.

    Notes
    -----
    The function also prints bounding boxes, areas, and centroid values.
    """

    fruits = {

        "Apple": {

            "blob": apple_blob,

            "colour": "red"

        },

        "Capsicum": {

            "blob": capsicum_blob,

            "colour": "lime"

        },

        "Lemon": {

            "blob": lemon_blob,

            "colour": "cyan"

        }

    }

    # Create figure

    fig, ax = plt.subplots(figsize=(13, 9))

    ax.imshow(image)

    results = {}

    for name, item in fruits.items():

        blob = item["blob"]

        colour = item["colour"]

        if blob is None or len(blob) == 0:

            print(f"{name}: Detection failed.")

            results[name] = None

            continue

        # Ensure correct NumPy array format

        blob = np.asarray(blob, dtype=int)

        if blob.ndim != 2 or blob.shape[1] != 2:

            raise ValueError(

                f"{name}: Expected blob shape (N, 2), "

                f"received {blob.shape}"

            )

        # Calculate final blob features

        features = calculate_blob_features(blob)

        results[name] = features

        # Extract bounding box

        xmin, ymin, xmax, ymax = features["bbox"]

        # Bounding box midpoint

        xb, yb = features["midpoint"]

        # Moment centroid

        xm, ym = features["moment"]

        # Draw bounding box

        rectangle = plt.Rectangle(

            (xmin - 0.5, ymin - 0.5),

            xmax - xmin + 1,

            ymax - ymin + 1,

            fill=False,

            edgecolor=colour,

            linewidth=2.5

        )

        ax.add_patch(rectangle)

        # Draw bounding box midpoint (X)

        ax.plot(

            xb, yb,

            marker="x",

            color="white",

            markeredgewidth=2.5,

            markersize=12,

            linestyle="None"

        )

        # Draw moment centroid (Circle)

        ax.plot(

            xm, ym,

            marker="o",

            color=colour,

            markeredgecolor="black",

            markeredgewidth=1.2,

            markersize=8,

            linestyle="None"

        )

        # Fruit label

        ax.text(

            xmin,

            max(12, ymin - 10),

            name,

            color="white",

            fontsize=12,

            fontweight="bold",

            bbox={

                "facecolor": colour,

                "alpha": 0.80,

                "edgecolor": "none",

                "boxstyle": "round,pad=0.3"

            }

        )

    # Figure formatting

    ax.set_title(

        f"Task 1 - Fruit Detection Results | Dark Factor = {darkFactor:.2f}, Mirror = {mirror}", 

        fontsize=17,

        pad=15

    )

    ax.axis("off")

    # Legend for centroid symbols

    legend_items = [

        Line2D(

            [0], [0],

            marker="x",

            color="white",

            markeredgecolor="black",

            markeredgewidth=2,

            linestyle="None",

            markersize=11,

            label="Bounding Box Midpoint"

        ),

        Line2D(

            [0], [0],

            marker="o",

            color="gray",

            markeredgecolor="black",

            linestyle="None",

            markersize=8,

            label="Moment Centroid"

        )

    ]

    ax.legend(

        handles=legend_items,

        loc="upper right",

        fontsize=10

    )

    plt.tight_layout()

    # Save high-resolution figure

    if save_path is not None:

        fig.savefig(

            save_path,

            dpi=300,

            bbox_inches="tight"

        )

        print(f"\nFigure saved to: {save_path}")

    plt.show()

    # Print numerical results

    print("\n========== FINAL DETECTION RESULTS ==========")

    for name, features in results.items():

        if features is None:

            print(f"\n{name}: Detection failed")

            continue

        print(f"\n{name}")

        print(f"  Area: {features['area']} pixels")

        print(f"  Bounding Box: {features['bbox']}")

        print(

            "  Bounding Box Midpoint: "

            f"({features['midpoint'][0]:.2f}, "

            f"{features['midpoint'][1]:.2f})"

        )

        print(

            "  Moment Centroid: "

            f"({features['moment'][0]:.2f}, "

            f"{features['moment'][1]:.2f})"

        )

        print(

            f"  Centroid Distance: "

            f"{features['distance']:.2f} pixels"

        )

    print("\n=============================================")

    return results

# %% 11. Main Pipeline

def main():

    """Execute the fruit detection pipeline and final visualisation.

    The current run uses the configured darkFactor and mirror. It preprocesses
    the input, creates colour masks, processes the three fruit candidates,
    prints final features, and saves the combined detection figure.

    Returns
    -------
    None
        Produces the final figure, console output, and the saved image.

    Notes
    -----
    This function uses the module-level image and output directory.
    """

    # Step 1: Preprocessing

    fruit_processed = preprocessing(

        darkFactor, mirror

    )

    # Step 2: Colour Segmentation

    apple_mask, capsicum_mask, lemon_mask = (

        colour_segmentation(fruit_processed)

    )

    # Step 3: Blob Detection + Region Growing

    apple_blob = (

        process_mask(apple_mask, fruit_processed, "Apple", mirror)

    )

    capsicum_blob = (

        process_mask(capsicum_mask, fruit_processed, "Capsicum", mirror)

    )

    lemon_blob = (

        process_mask(lemon_mask, fruit_processed, "Lemon", mirror)

    )

    # Step 4: Final Results

    final_results = visualise_results(

        fruit_processed,

        apple_blob,

        capsicum_blob,

        lemon_blob,

        darkFactor,

        mirror,

        save_path=folder / f"Task1_Final_Results_{darkFactor:.2f}_{mirror}.png"

)

if __name__ == "__main__":
    main()
