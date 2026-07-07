"""Generates synthetic chart images with known ground-truth data for tests."""
import io

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _fig_to_bgr_array(fig) -> np.ndarray:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    file_bytes = np.frombuffer(buf.read(), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return image


def make_line_chart():
    x = np.linspace(0, 10, 11)
    y = x * 8 + 5  # simple known line: y = 8x + 5, range 5..85

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, y, color="blue", linewidth=2)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Time (months)")
    ax.set_ylabel("Value")

    image = _fig_to_bgr_array(fig)
    truth = {
        "x_range": (0.0, 10.0),
        "y_range": (0.0, 100.0),
        "series": [
            {"name": "Series A", "color": "blue", "points": list(zip(x.tolist(), y.tolist()))}
        ],
    }
    return image, truth


def make_km_chart():
    """Two-arm step-function survival curve with censoring ticks."""
    t = np.array([0, 2, 4, 6, 8, 10, 12])
    surv_a = np.array([1.0, 0.95, 0.85, 0.70, 0.60, 0.55, 0.50])
    surv_b = np.array([1.0, 0.90, 0.70, 0.50, 0.35, 0.25, 0.20])
    censor_t_a = np.array([5, 9])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.step(t, surv_a, where="post", color="red", linewidth=2, label="Arm A")
    ax.step(t, surv_b, where="post", color="green", linewidth=2, label="Arm B")
    ax.plot(censor_t_a, np.interp(censor_t_a, t, surv_a), "+", color="red", markersize=10)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Time (months)")
    ax.set_ylabel("Survival probability")
    ax.legend(loc="upper right")

    image = _fig_to_bgr_array(fig)
    truth = {
        "x_range": (0.0, 12.0),
        "y_range": (0.0, 1.0),
        "series": [
            {"name": "Arm A", "color": "red", "points": list(zip(t.tolist(), surv_a.tolist()))},
            {"name": "Arm B", "color": "green", "points": list(zip(t.tolist(), surv_b.tolist()))},
        ],
    }
    return image, truth


def make_scatter_chart():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 10, size=15)
    y = 3 * x + rng.normal(0, 1, size=15)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(x, y, color="purple")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 40)
    ax.set_xlabel("Dose (mg)")
    ax.set_ylabel("Response")

    image = _fig_to_bgr_array(fig)
    truth = {
        "x_range": (0.0, 10.0),
        "y_range": (0.0, 40.0),
        "series": [{"name": "Series A", "color": "purple", "points": list(zip(x.tolist(), y.tolist()))}],
    }
    return image, truth


def make_bar_chart():
    categories_x = np.array([1, 2, 3, 4])
    heights = np.array([12.0, 25.0, 18.0, 30.0])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(categories_x, heights, color="teal")
    ax.set_ylim(0, 40)
    ax.set_xlabel("Group")
    ax.set_ylabel("Count")

    image = _fig_to_bgr_array(fig)
    truth = {
        "x_range": (0.5, 4.5),
        "y_range": (0.0, 40.0),
        "series": [
            {"name": "Series A", "color": "teal", "points": list(zip(categories_x.tolist(), heights.tolist()))}
        ],
    }
    return image, truth
