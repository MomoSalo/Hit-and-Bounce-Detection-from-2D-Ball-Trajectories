import os
import json
import numpy as np

MIN_VISIBLE_FRAMES = 30

def compute_event_percentage(folder_path):
    total_visible = 0
    total_events = 0
    total_hit = 0
    total_bounce = 0

    for filename in os.listdir(folder_path):
        if not filename.endswith(".json"):
            continue

        with open(os.path.join(folder_path, filename), "r") as f:
            data = json.load(f)

        for d in data.values():
            if not d.get("visible"):
                continue

            total_visible += 1

            if d.get("action") in ["hit", "bounce"]:
                total_events += 1

            if d.get("action") == "hit":
                total_hit += 1

            elif d.get("action") == "bounce":
                total_bounce += 1

    if total_visible == 0:
        raise ValueError("No visible frames found.")

    return {
        "percent_events": 100 * total_events / total_visible,
        "percent_hit": 100 * total_hit / total_visible,
        "percent_bounce": 100 * total_bounce / total_visible,
        "total_visible_frames": total_visible
    }



def compute_T_bounce(folder_path, q=99, method="percentile"):
    """
    Compute T_bounce from AIR frames only.

    Parameters
    ----------
    folder_path : str
        Path to folder containing JSON files.
    method : str
        "percentile" or "mad"
    q : int
        Percentile value if method == "percentile"

    Returns
    -------
    T_bounce : float
    """

    air_delta_dy = []

    for filename in os.listdir(folder_path):
        if not filename.endswith(".json"):
            continue

        with open(os.path.join(folder_path, filename), "r") as f:
            data = json.load(f)

        # Convert frame keys to int and sort
        data = {int(k): v for k, v in data.items()}
        frames = sorted(data.keys())

        # Extract visible x, y and labels
        x_vals, y_vals, labels = [], [], []

        for f in frames:
            d = data[f]
            if d.get("visible") and d.get("x") is not None and d.get("y") is not None:
                x_vals.append(d["x"])
                y_vals.append(d["y"])
                labels.append(d.get("action"))

        if len(x_vals) < MIN_VISIBLE_FRAMES:
            continue

        x_vals = np.array(x_vals, dtype=float)
        y_vals = np.array(y_vals, dtype=float)

        # First derivative
        dy = np.diff(y_vals)

        # Second derivative (jump in slope)
        delta_dy = np.diff(dy)

        # Align labels with delta_dy
        labels_delta = labels[2:]

        for i, label in enumerate(labels_delta):
            if label == "air":
                air_delta_dy.append(abs(delta_dy[i]))

    air_delta_dy = np.array(air_delta_dy)

    if len(air_delta_dy) == 0:
        raise ValueError("No AIR delta_dy found. Cannot compute T_bounce.")

    # ---- Compute threshold ----
    if method == "percentile":
        T_bounce = np.percentile(air_delta_dy, q)

    elif method == "mad":
        med = np.median(air_delta_dy)
        mad = np.median(np.abs(air_delta_dy - med))
        T_bounce = med + 7 * mad

    else:
        raise ValueError("Unknown method. Use 'percentile' or 'mad'.")

    return T_bounce
