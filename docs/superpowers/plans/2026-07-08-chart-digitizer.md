# Chart Digitizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and deploy a hosted web app that automatically digitizes data points from uploaded chart images (line/scatter/bar/Kaplan-Meier), with a visual drag-to-correct review step and CSV/Excel export.

**Architecture:** Python FastAPI backend (OpenCV geometry detection + Tesseract OCR + color-based curve tracing, stateless, deployed to Render as a Docker service) and a React+TypeScript frontend (Vite, Konva canvas for the review/edit step, deployed to Vercel). No database, no external LLM API. See `docs/superpowers/specs/2026-07-07-chart-digitizer-design.md` for the full design rationale.

**Tech Stack:** Python 3.11, FastAPI, opencv-python-headless, pytesseract + system `tesseract-ocr`, numpy, openpyxl, itsdangerous (session cookies), pytest, matplotlib (test fixture generation) · React 18, TypeScript, Vite, react-konva, Vitest + React Testing Library

---

## File Structure

**Backend** (`backend/`):
- `backend/app/main.py` — FastAPI app, route registration, CORS
- `backend/app/auth.py` — password check, signed session cookie issue/verify, auth dependency
- `backend/app/pipeline/geometry.py` — plot box + axis line + tick pixel-position detection
- `backend/app/pipeline/ocr.py` — axis tick text, axis titles, legend text extraction via Tesseract
- `backend/app/pipeline/calibration.py` — pixel↔data transform fitting (linear + log)
- `backend/app/pipeline/classify.py` — chart-type heuristic (line/scatter/bar/KM)
- `backend/app/pipeline/curves.py` — per-series color mask isolation + point/curve extraction per chart type
- `backend/app/pipeline/overlay.py` — draws detected points/curves back onto the image for QC
- `backend/app/pipeline/pipeline.py` — orchestrates the full extraction, returns structured result
- `backend/app/export.py` — CSV/Excel generation from corrected series data
- `backend/app/models.py` — Pydantic request/response models
- `backend/requirements.txt`, `backend/Dockerfile`
- `backend/tests/fixtures.py` — matplotlib-based synthetic chart image generator used across tests
- `backend/tests/test_*.py` — one test file per pipeline module

**Frontend** (`frontend/`):
- `frontend/src/pages/Login.tsx`
- `frontend/src/pages/Upload.tsx`
- `frontend/src/pages/Review.tsx`
- `frontend/src/components/ChartCanvas.tsx` — Konva image + draggable point/axis-marker overlay
- `frontend/src/components/DataTable.tsx`
- `frontend/src/api.ts` — typed fetch wrappers for backend endpoints
- `frontend/src/types.ts`
- `frontend/src/App.tsx`, `frontend/src/main.tsx`

---

## Task 1: Backend scaffolding

**Files:**
- Create: `backend/requirements.txt`
- Create: `backend/app/__init__.py`
- Create: `backend/app/main.py`
- Test: `backend/tests/test_main.py`

- [ ] **Step 1: Create the directory structure and requirements file**

```bash
mkdir -p backend/app/pipeline backend/tests
```

`backend/requirements.txt`:
```
fastapi==0.115.0
uvicorn[standard]==0.30.6
opencv-python-headless==4.10.0.84
numpy==1.26.4
pytesseract==0.3.13
Pillow==10.4.0
openpyxl==3.1.5
itsdangerous==2.2.0
python-multipart==0.0.9
pytest==8.3.3
httpx==0.27.2
matplotlib==3.9.2
```

- [ ] **Step 2: Write the failing test for a health check endpoint**

`backend/tests/test_main.py`:
```python
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_check_returns_ok():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
```

- [ ] **Step 3: Run test to verify it fails**

Run (from `backend/`): `pytest tests/test_main.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app'` or import error, since `app/main.py` doesn't exist yet.

- [ ] **Step 4: Create empty package init and the FastAPI app**

`backend/app/__init__.py`: (empty file)

`backend/app/main.py`:
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Chart Digitizer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health_check():
    return {"status": "ok"}
```

- [ ] **Step 5: Install dependencies and run test to verify it passes**

Run: `pip install -r backend/requirements.txt`
Run: `cd backend && pytest tests/test_main.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add backend/requirements.txt backend/app/__init__.py backend/app/main.py backend/tests/test_main.py
git commit -m "feat: scaffold FastAPI backend with health check"
```

---

## Task 2: Synthetic chart test-fixture generator

Every later pipeline task needs known-ground-truth chart images to test against. Build this once, reuse everywhere.

**Files:**
- Create: `backend/tests/fixtures.py`
- Test: `backend/tests/test_fixtures.py`

- [ ] **Step 1: Write the failing test**

`backend/tests/test_fixtures.py`:
```python
import numpy as np
from tests.fixtures import make_line_chart


def test_make_line_chart_returns_image_and_ground_truth():
    image, truth = make_line_chart()

    assert isinstance(image, np.ndarray)
    assert image.ndim == 3  # H, W, 3 (BGR)
    assert truth["x_range"] == (0.0, 10.0)
    assert truth["y_range"] == (0.0, 100.0)
    assert len(truth["series"]) == 1
    assert truth["series"][0]["name"] == "Series A"
    assert len(truth["series"][0]["points"]) == 11
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/test_fixtures.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tests.fixtures'`

- [ ] **Step 3: Write the fixture generator**

`backend/tests/fixtures.py`:
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && pytest tests/test_fixtures.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/tests/fixtures.py backend/tests/test_fixtures.py
git commit -m "test: add synthetic chart image generator for pipeline tests"
```

---

## Task 3: Plot box and axis tick pixel-position detection

**Files:**
- Create: `backend/app/pipeline/__init__.py`
- Create: `backend/app/pipeline/geometry.py`
- Test: `backend/tests/test_geometry.py`

- [ ] **Step 1: Write the failing test**

`backend/tests/test_geometry.py`:
```python
from app.pipeline.geometry import detect_plot_box, detect_tick_positions
from tests.fixtures import make_line_chart


def test_detect_plot_box_finds_axes_rectangle():
    image, _ = make_line_chart()
    box = detect_plot_box(image)

    h, w = image.shape[:2]
    assert 0 <= box.left < box.right <= w
    assert 0 <= box.top < box.bottom <= h
    # plot area should be a substantial, sane portion of the image
    assert (box.right - box.left) > w * 0.4
    assert (box.bottom - box.top) > h * 0.4


def test_detect_tick_positions_finds_expected_count_on_each_axis():
    image, truth = make_line_chart()
    box = detect_plot_box(image)

    x_ticks = detect_tick_positions(image, box, axis="x")
    y_ticks = detect_tick_positions(image, box, axis="y")

    # matplotlib default gives ~6 ticks per axis for these ranges; allow slack
    assert 4 <= len(x_ticks) <= 8
    assert 4 <= len(y_ticks) <= 8
    # x ticks should be sorted left-to-right, y ticks top-to-bottom
    assert x_ticks == sorted(x_ticks)
    assert y_ticks == sorted(y_ticks)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/test_geometry.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.pipeline'`

- [ ] **Step 3: Implement plot box and tick detection**

`backend/app/pipeline/__init__.py`: (empty file)

`backend/app/pipeline/geometry.py`:
```python
"""Detects the plot bounding box and axis tick pixel positions using classic
edge/line detection. Charts are assumed to have a visible left/bottom axis
spine (true for matplotlib default style and the vast majority of published
figures)."""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PlotBox:
    left: int
    top: int
    right: int
    bottom: int


def _to_gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def detect_plot_box(image: np.ndarray) -> PlotBox:
    gray = _to_gray(image)
    edges = cv2.Canny(gray, 50, 150)

    h, w = gray.shape
    min_line_length = min(h, w) * 0.3
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=100,
        minLineLength=min_line_length, maxLineGap=10,
    )
    if lines is None:
        raise ValueError("No axis lines detected in image")

    horizontals, verticals = [], []
    for x1, y1, x2, y2 in lines[:, 0]:
        if abs(y1 - y2) < 3 and abs(x1 - x2) > min_line_length:
            horizontals.append((y1 + y2) / 2)
        elif abs(x1 - x2) < 3 and abs(y1 - y2) > min_line_length:
            verticals.append((x1 + x2) / 2)

    if not horizontals or not verticals:
        raise ValueError("Could not find both horizontal and vertical axis lines")

    # bottom axis = lowest horizontal line, left axis = leftmost vertical line,
    # top/right bounds taken from the extreme opposite lines found (handles
    # boxed plots) or fall back to image edges (handles open-spine plots).
    bottom = int(max(horizontals))
    top_candidates = [y for y in horizontals if y < bottom - 10]
    top = int(min(top_candidates)) if top_candidates else int(h * 0.05)

    left = int(min(verticals))
    right_candidates = [x for x in verticals if x > left + 10]
    right = int(max(right_candidates)) if right_candidates else int(w * 0.95)

    return PlotBox(left=left, top=top, right=right, bottom=bottom)


def detect_tick_positions(image: np.ndarray, box: PlotBox, axis: str) -> list[int]:
    """Finds tick mark pixel positions along the given axis by looking for
    short perpendicular strokes just outside the plot box spine."""
    gray = _to_gray(image)
    edges = cv2.Canny(gray, 50, 150)

    if axis == "x":
        # look in a thin strip just below the bottom spine
        strip = edges[box.bottom + 1: box.bottom + 8, box.left:box.right]
        column_sums = strip.sum(axis=0)
        positions_local = _find_peak_centers(column_sums)
        return [box.left + p for p in positions_local]
    elif axis == "y":
        # look in a thin strip just left of the left spine
        strip = edges[box.top:box.bottom, max(0, box.left - 8):box.left - 1]
        row_sums = strip.sum(axis=1)
        positions_local = _find_peak_centers(row_sums)
        return [box.top + p for p in positions_local]
    else:
        raise ValueError(f"Unknown axis: {axis}")


def _find_peak_centers(signal: np.ndarray, min_gap: int = 5) -> list[int]:
    """Groups consecutive nonzero indices in a 1D signal into clusters and
    returns each cluster's center — used to turn a strip of tick-mark pixels
    into one position per tick."""
    nonzero = np.where(signal > 0)[0]
    if len(nonzero) == 0:
        return []

    clusters: list[list[int]] = [[int(nonzero[0])]]
    for idx in nonzero[1:]:
        if idx - clusters[-1][-1] <= min_gap:
            clusters[-1].append(int(idx))
        else:
            clusters.append([int(idx)])

    return [int(np.mean(cluster)) for cluster in clusters]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && pytest tests/test_geometry.py -v`
Expected: PASS. If tick counts fall outside the 4-8 range, print `x_ticks`/`y_ticks` and adjust `min_gap` in `_find_peak_centers` or the strip thickness (`+8`) — matplotlib tick mark length varies slightly with DPI.

- [ ] **Step 5: Commit**

```bash
git add backend/app/pipeline/__init__.py backend/app/pipeline/geometry.py backend/tests/test_geometry.py
git commit -m "feat: detect plot box and axis tick pixel positions"
```

---
