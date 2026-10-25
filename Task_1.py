# CODE MAP of Task_1.py
# 1. Imports and Configuration                             Import required libraries and define settings
# 2. Image Loading                                         Load Fruits.jpg
# 3. Preprocessing                                         Apply brightness scaling and horizontal mirroring
# 4. Colour Segmentation                                   Generate fruit masks using normalised RGB
# 5. Connected Component Analysis                          Find connected foreground components
# 6. Blob Selection                                        Filter and rank candidate fruit blobs
# 7. Region Growing and Diagnostic Statistics              Refine fruit regions and report diagnostics
# 8. Geometric Feature Extraction                          Compute bounding boxes, centroids and areas
# 9. Experimental Utility: Candidate Blob Visualisation    Display ranked blob candidates
# 10. Experimental Utility: Pixel Growth Visualisation     Display seed pixels and newly grown pixels
# 11. Experimental Utility: Bounding Box Comparison        Compare bounding boxes before and after growth
# 12. Fruit Processing Pipeline                            Run detection and feature extraction per fruit
# 13. Final Results Visualisation                          Display and save final detection results
# 14. Main Pipeline                                        Execute the complete processing pipeline
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
    The function also displays the original image and the three masks.
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

    masks = [apple_mask, capsicum_mask, lemon_mask]

    names = ["Apple", "Capsicum", "Lemon"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(image)

    axes[0].set_title("Original Image")

    for i in range(3):

        axes[i + 1].imshow(masks[i], cmap="gray")

        axes[i + 1].set_title(names[i] + " Mask")

    for ax in axes:

        ax.axis("off")

    plt.tight_layout()

    plt.show()

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

# %% 7. Region Growing and Diagnostic Statistics

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
    All original seed pixels remain included. Diagnostic counters, bounding
    boxes, and directional statistics are printed for experimental analysis.
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

        eligible = (

            allowed &

            (colour_distance <= colour_distance_threshold)

        )

    else:

        eligible = allowed.copy()

    # Diagnostic counters

    rejected_colour = 0

    rejected_similarity = 0

    rejected_colour_mask = np.zeros(

        (H, W), dtype=bool

    )

    rejected_similarity_mask = np.zeros(

        (H, W), dtype=bool

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

                rejected_colour += 1

                rejected_colour_mask[ny, nx] = True

                continue

            # Similarity condition

            if (

                colour_distance_threshold is not None

                and

                colour_distance[ny, nx] >

                colour_distance_threshold

            ):

                rejected_similarity += 1

                rejected_similarity_mask[ny, nx] = True

                continue

            # Accept pixel

            grown[ny, nx] = True

            queue.append((ny, nx))

            area += 1

            if area >= max_area:

                break

    # Step 8: Diagnostics

    grown_blob = np.argwhere(grown)

    grown_ys = grown_blob[:, 0]

    grown_xs = grown_blob[:, 1]

    after_bbox = (

        int(grown_xs.min()),

        int(grown_ys.min()),

        int(grown_xs.max()),

        int(grown_ys.max())

    )

    # Regions immediately below / right of seed

    below = np.zeros((H, W), dtype=bool)

    if ymax < bottom:

        below[

            ymax + 1:bottom + 1,

            xmin:xmax + 1

        ] = True

    right_side = np.zeros((H, W), dtype=bool)

    if xmax < right:

        right_side[

            ymin:ymax + 1,

            xmax + 1:right + 1

        ] = True

    print(f"\n========== {fruit_name} DIAGNOSTICS ==========")

    print("Seed area:", len(seed_blob))

    print("Final area:", area)

    print("Area increase:", area - len(seed_blob))

    print("\nMaximum allowed area:", max_area)

    print("Area limit reached:", area >= max_area)

    print("Queue exhausted:", len(queue) == 0)

    print("\nRejected neighbour checks:")

    print("  Colour threshold:", rejected_colour)

    print("  Colour similarity:", rejected_similarity)

    print("\nUnique rejected pixels:")

    print(

        "  Colour:",

        np.count_nonzero(rejected_colour_mask)

    )

    print(

        "  Similarity:",

        np.count_nonzero(rejected_similarity_mask)

    )

    print("\nBounding boxes:")

    print(

        "  Before:",

        (xmin, ymin, xmax, ymax)

    )

    print("  After: ", after_bbox)

    print("\nSearch boundaries:")

    print(

        f"  Left: {left}, Right: {right}"

    )

    print(

        f"  Top: {top}, Bottom: {bottom}"

    )

    print("\nDirectional diagnostics:")

    print(

        "  Eligible pixels below seed:",

        np.count_nonzero(eligible & below)

    )

    print(

        "  Grown pixels below seed:   ",

        np.count_nonzero(grown & below)

    )

    print(

        "  Eligible pixels right of seed:",

        np.count_nonzero(eligible & right_side)

    )

    print(

        "  Grown pixels right of seed:   ",

        np.count_nonzero(grown & right_side)

    )

    print("==========================================\n")

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

# %% 9. Experimental Utility: Candidate Blob Visualisation

def visualise_blobs(image, candidates, name):

    """Display up to ten ranked blob candidates on the input image.

    The highest-scoring candidate is outlined in green and remaining
    candidates in red. Each rectangle is annotated with its rank and score.

    Parameters
    ----------
    image : numpy.ndarray
        RGB image used as the background.
    candidates : list of tuple
        Ranked (score, blob) pairs, with blob coordinates in (y, x) order.
    name : str
        Fruit name used in the plot title.

    Returns
    -------
    None
        Displays a Matplotlib figure.

    Notes
    -----
    Experimental visualisation; it does not change detection results.
    """

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.imshow(image)

    for i, (score, blob) in enumerate(candidates[:10]):

        ys = blob[:, 0]

        xs = blob[:, 1]

        xmin, xmax = xs.min(), xs.max()

        ymin, ymax = ys.min(), ys.max()

        # Green for best blob, red for others

        colour = "lime" if i == 0 else "red"

        rect = plt.Rectangle(

            (xmin - 0.5, ymin - 0.5),

            xmax - xmin + 1,

            ymax - ymin + 1,

            fill=False,

            edgecolor=colour,

            linewidth=2

        )

        ax.add_patch(rect)

        ax.text(

            xmin, ymin,

            f"{i + 1}: {score:.0f}",

            color="blue",

            bbox=dict(facecolor="white", alpha=0.8)

        )

    ax.set_title(f"{name} - Filtered Candidate Blobs")

    ax.axis("off")

    plt.show()

# The following visualisation functions are auxiliary experimental tools.
# They remain part of the original execution flow in main().

# %% 10. Experimental Utility: Pixel Growth Visualisation

def visualise_growth_pixels(

    image, seed_blob, grown_blob, name, padding=40

):

    """Compare original seed pixels with pixels added by region growing.

    A cropped two-panel figure shows the original image and an overlay.
    Seed pixels are tinted green and newly accepted pixels red. The function
    also prints the seed area, added area, final area, and percentage growth.

    Parameters
    ----------
    image : numpy.ndarray
        Input RGB image.
    seed_blob : numpy.ndarray
        Original seed coordinates in (y, x) order.
    grown_blob : numpy.ndarray
        Final region coordinates in (y, x) order.
    name : str
        Fruit name used in the figure and statistics.
    padding : int, optional
        Additional crop margin in pixels (default 40).

    Returns
    -------
    None
        Displays a figure and prints experimental pixel statistics.

    Notes
    -----
    This is an auxiliary diagnostic, not a separate detection stage.
    """

    if seed_blob is None or grown_blob is None:

        print(f"{name}: No valid blob to visualise.")

        return

    H, W = image.shape[:2]

    # Create binary masks

    seed_mask = np.zeros((H, W), dtype=bool)

    grown_mask = np.zeros((H, W), dtype=bool)

    seed_mask[

        seed_blob[:, 0], seed_blob[:, 1]

    ] = True

    grown_mask[

        grown_blob[:, 0], grown_blob[:, 1]

    ] = True

    # Identify newly added pixels

    new_pixels = grown_mask & ~seed_mask

    # Create RGB overlay

    overlay = image.astype(float).copy()

    # Seed pixels: Green

    overlay[seed_mask] = (

        0.5 * overlay[seed_mask] +

        0.5 * np.array([0, 255, 0])

    )

    # Newly grown pixels: Red

    overlay[new_pixels] = (

        0.4 * overlay[new_pixels] +

        0.6 * np.array([255, 0, 0])

    )

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    # Determine region for zooming

    all_pixels = np.vstack([seed_blob, grown_blob])

    ymin = int(all_pixels[:, 0].min())

    ymax = int(all_pixels[:, 0].max())

    xmin = int(all_pixels[:, 1].min())

    xmax = int(all_pixels[:, 1].max())

    x1 = max(0, xmin - padding)

    x2 = min(W, xmax + padding + 1)

    y1 = max(0, ymin - padding)

    y2 = min(H, ymax + padding + 1)

    # Visualisation

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].imshow(image[y1:y2, x1:x2])

    axes[0].set_title(f"{name} - Original")

    axes[1].imshow(overlay[y1:y2, x1:x2])

    axes[1].set_title(f"{name} - Region Growing Pixels")

    for ax in axes:

        ax.axis("off")

    plt.tight_layout()

    plt.show()

    # Print statistics

    seed_area = np.count_nonzero(seed_mask)

    new_area = np.count_nonzero(new_pixels)

    final_area = np.count_nonzero(grown_mask)

    print(f"\n{name} Pixel Analysis:")

    print(f"Seed area: {seed_area}")

    print(f"New pixels: {new_area}")

    print(f"Final area: {final_area}")

    print(f"Growth percentage: {new_area / seed_area * 100:.2f}%")

# %% 11. Experimental Utility: Bounding Box Comparison

def visualise_growth(image, candidates, grown_blob, name):

    """Compare bounding boxes before and after region growing.

    The original selected candidate is shown with a red bounding box and
    the final grown region with a green bounding box. Numerical extensions
    on each side are printed to the terminal.

    Parameters
    ----------
    image : numpy.ndarray
        RGB image used as the background.
    candidates : list of tuple
        Ranked (score, blob) pairs; the first supplies the original seed.
    grown_blob : numpy.ndarray
        Final region pixel coordinates in (y, x) order.
    name : str
        Fruit name for the title and printed output.

    Returns
    -------
    None
        Displays a Matplotlib figure and prints bounding-box differences.

    Notes
    -----
    Experimental comparison only; no segmentation is changed.
    """

    if candidates is None or len(candidates) == 0:

        print(f"{name}: No seed blob found.")

        return

    if grown_blob is None or len(grown_blob) == 0:

        print(f"{name}: No grown blob found.")

        return

    # Extract original seed blob

    # candidates contains (score, blob) pairs

    seed_blob = np.asarray(candidates[0][1], dtype=int)

    grown_blob = np.asarray(grown_blob, dtype=int)

    # Validate blob coordinates

    if (

        seed_blob.ndim != 2 or seed_blob.shape[1] != 2 or

        grown_blob.ndim != 2 or grown_blob.shape[1] != 2

    ):

        raise ValueError(

            f"{name}: Blob coordinates must have shape (N, 2). "

            f"Seed: {seed_blob.shape}, "

            f"Grown: {grown_blob.shape}"

        )

    # Calculate bounding boxes

    # Blob coordinates are stored as (y, x)

    seed_ys = seed_blob[:, 0]

    seed_xs = seed_blob[:, 1]

    grown_ys = grown_blob[:, 0]

    grown_xs = grown_blob[:, 1]

    before_bbox = (

        int(seed_xs.min()),

        int(seed_ys.min()),

        int(seed_xs.max()),

        int(seed_ys.max())

    )

    after_bbox = (

        int(grown_xs.min()),

        int(grown_ys.min()),

        int(grown_xs.max()),

        int(grown_ys.max())

    )

    # Create figure

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.imshow(image)

    # Before growing - Red bounding box

    xmin, ymin, xmax, ymax = before_bbox

    before_rect = plt.Rectangle(

        (xmin - 0.5, ymin - 0.5),

        xmax - xmin + 1,

        ymax - ymin + 1,

        fill=False,

        edgecolor="red",

        linewidth=2.5,

        label="Before Growing"

    )

    ax.add_patch(before_rect)

    # After growing - Green bounding box

    xmin, ymin, xmax, ymax = after_bbox

    after_rect = plt.Rectangle(

        (xmin - 0.5, ymin - 0.5),

        xmax - xmin + 1,

        ymax - ymin + 1,

        fill=False,

        edgecolor="lime",

        linewidth=2.5,

        label="After Growing"

    )

    ax.add_patch(after_rect)

    # Figure settings

    ax.set_title(

        f"{name} - Region Growing",

        fontsize=16

    )

    ax.legend(loc="upper right")

    ax.axis("off")

    plt.tight_layout()

    plt.show()

    # Print comparison

    bx1, by1, bx2, by2 = before_bbox

    ax1, ay1, ax2, ay2 = after_bbox

    print(f"\n========== {name} BOUNDING BOX ==========")

    print("Before Growing:", before_bbox)

    print("After Growing: ", after_bbox)

    print("\nBounding Box Expansion:")

    print(f"  Left:   {bx1 - ax1} pixels")

    print(f"  Right:  {ax2 - bx2} pixels")

    print(f"  Top:    {by1 - ay1} pixels")

    print(f"  Bottom: {ay2 - by2} pixels")

    print("=========================================\n")

# %% 12. Fruit Processing Pipeline

def process_mask(mask, image, fruit_name, mirror=False):

    # Connected Components

    """Run component detection, selection, and refinement for one fruit.

    The binary mask is separated into four-connected components. The best
    candidate is selected using geometric scoring, then expanded with the
    existing fruit-specific region-growing parameters. Geometric features
    before and after expansion are printed.

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
    after : dict or None
        Final geometric features from calculate_blob_features().
    candidates : list of tuple
        Ranked (score, blob) candidate pairs.
    best_blob : numpy.ndarray or None
        Original seed component before region growing.

    Notes
    -----
    When no candidate qualifies, the function returns
    (None, None, candidates, None).
    """

    components = connected_components(mask)

    # Blob Selection

    best_blob, candidates = select_best_blob(

        components, mask.shape

    )

    if best_blob is None:

        print(f"Warning: {fruit_name} not detected.")

        return None, None, candidates, None

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

    # Calculate features before and after growing

    before = calculate_blob_features(best_blob)

    after = calculate_blob_features(grown_blob)

    print(f"\n{fruit_name}:")

    print(f"Area before: {before['area']}")

    print(f"Area after:  {after['area']}")

    print(f"BBox before: {before['bbox']}")

    print(f"BBox after:  {after['bbox']}")

    return grown_blob, after, candidates, best_blob

# %% 13. Final Results Visualisation

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

# %% 14. Main Pipeline

def main():

    """Execute the fruit detection pipeline and all configured visualisations.

    The current run uses darkFactor = 1.0 and mirror = False. It preprocesses
    the input, creates colour masks, processes the three fruit candidates,
    displays pixel-growth and bounding-box diagnostic figures, prints final
    features, and saves the combined detection figure.

    Returns
    -------
    None
        Produces figures, console output, and the final saved image.

    Notes
    -----
    The diagnostic visualisations remain enabled exactly as in the source.
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

    apple_blob, apple_features, apple_candidates, apple_seed = (

        process_mask(apple_mask, fruit_processed, "Apple", mirror)

    )

    capsicum_blob, capsicum_features, capsicum_candidates, capsicum_seed = (

        process_mask(capsicum_mask, fruit_processed, "Capsicum", mirror)

    )

    lemon_blob, lemon_features, lemon_candidates, lemon_seed = (

        process_mask(lemon_mask, fruit_processed, "Lemon", mirror)

    )

    # Step 4: Bounding Box Comparison

    visualise_growth_pixels(

        fruit_processed,

        apple_seed,

        apple_blob,

        "Apple"

    )

    visualise_growth_pixels(

        fruit_processed,

        capsicum_seed,

        capsicum_blob,

        "Capsicum"

    )

    visualise_growth_pixels(

        fruit_processed,

        lemon_seed,

        lemon_blob,

        "Lemon"

    )

    visualise_growth(

    fruit_processed,

    apple_candidates,

    apple_blob,

    "Apple"

)

    visualise_growth(

        fruit_processed,

        capsicum_candidates,

        capsicum_blob,

        "Capsicum"

    )

    visualise_growth(

        fruit_processed,

        lemon_candidates,

        lemon_blob,

        "Lemon"

    )

    # Step 6: Final Results

    print("\n===== FINAL RESULTS =====")

    print("Apple:", apple_features)

    print("Capsicum:", capsicum_features)

    print("Lemon:", lemon_features)

    # %% Final Experimental Visualisation

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
