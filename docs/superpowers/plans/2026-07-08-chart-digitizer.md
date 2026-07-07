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

## Task 4: OCR of axis tick labels and titles

**Prerequisite:** the `tesseract` binary must be installed locally to run these tests (`choco install tesseract` on Windows, or `apt-get install tesseract-ocr` on Linux/CI). This is separate from the `pytesseract` pip package, which is just a wrapper.

**Files:**
- Create: `backend/app/pipeline/ocr.py`
- Test: `backend/tests/test_ocr.py`

- [ ] **Step 1: Write the failing test**

`backend/tests/test_ocr.py`:
```python
from app.pipeline.geometry import detect_plot_box, detect_tick_positions
from app.pipeline.ocr import read_axis_tick_labels
from tests.fixtures import make_line_chart


def test_read_axis_tick_labels_reads_numeric_y_values():
    image, _ = make_line_chart()
    box = detect_plot_box(image)
    y_ticks = detect_tick_positions(image, box, axis="y")

    labels = read_axis_tick_labels(image, box, y_ticks, axis="y")

    assert len(labels) == len(y_ticks)
    numeric_values = [label.value for label in labels if label.value is not None]
    # y range is 0..100 in steps of 20 by default matplotlib behavior;
    # require most ticks to be read correctly (OCR isn't perfect)
    assert len(numeric_values) >= len(y_ticks) - 1
    assert numeric_values == sorted(numeric_values, reverse=True)  # top-to-bottom => descending
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/test_ocr.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.pipeline.ocr'`

- [ ] **Step 3: Implement OCR reading**

`backend/app/pipeline/ocr.py`:
```python
"""Reads axis tick label text/numbers, axis titles, and legend text from a
chart image using Tesseract OCR. Each region is cropped and upscaled before
OCR to maximize accuracy on small chart label text."""
import re
from dataclasses import dataclass

import cv2
import numpy as np
import pytesseract

from app.pipeline.geometry import PlotBox

_UPSCALE_FACTOR = 3
_NUMBER_PATTERN = re.compile(r"-?\d+\.?\d*")


@dataclass
class TickLabel:
    pixel_position: int
    text: str
    value: float | None


def _preprocess_for_ocr(crop: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    upscaled = cv2.resize(
        gray, None, fx=_UPSCALE_FACTOR, fy=_UPSCALE_FACTOR, interpolation=cv2.INTER_CUBIC
    )
    _, binary = cv2.threshold(upscaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def _parse_number(text: str) -> float | None:
    text = text.strip().replace(",", "")
    match = _NUMBER_PATTERN.search(text)
    if match is None:
        return None
    try:
        return float(match.group())
    except ValueError:
        return None


def read_axis_tick_labels(
    image: np.ndarray, box: PlotBox, tick_positions: list[int], axis: str
) -> list[TickLabel]:
    labels: list[TickLabel] = []
    h, w = image.shape[:2]

    for pos in tick_positions:
        if axis == "x":
            crop = image[box.bottom + 5: min(h, box.bottom + 35), max(0, pos - 30): pos + 30]
            config = "--psm 7"
        elif axis == "y":
            crop = image[max(0, pos - 12): pos + 12, max(0, box.left - 60): max(0, box.left - 5)]
            config = "--psm 7"
        else:
            raise ValueError(f"Unknown axis: {axis}")

        if crop.size == 0:
            labels.append(TickLabel(pixel_position=pos, text="", value=None))
            continue

        processed = _preprocess_for_ocr(crop)
        text = pytesseract.image_to_string(processed, config=config).strip()
        labels.append(TickLabel(pixel_position=pos, text=text, value=_parse_number(text)))

    return labels


def read_axis_title(image: np.ndarray, box: PlotBox, axis: str) -> str:
    h, w = image.shape[:2]
    if axis == "x":
        crop = image[min(h, box.bottom + 35): min(h, box.bottom + 65), box.left:box.right]
        processed = _preprocess_for_ocr(crop)
        return pytesseract.image_to_string(processed, config="--psm 7").strip()
    elif axis == "y":
        crop = image[box.top:box.bottom, max(0, box.left - 95): max(0, box.left - 60)]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        rotated = cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE)
        upscaled = cv2.resize(rotated, None, fx=_UPSCALE_FACTOR, fy=_UPSCALE_FACTOR, interpolation=cv2.INTER_CUBIC)
        _, binary = cv2.threshold(upscaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return pytesseract.image_to_string(binary, config="--psm 7").strip()
    else:
        raise ValueError(f"Unknown axis: {axis}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && pytest tests/test_ocr.py -v`
Expected: PASS. OCR accuracy is sensitive to crop padding — if it fails, print `labels` to see the raw `text` Tesseract returned per tick and adjust the crop box size/offset in `read_axis_tick_labels`.

- [ ] **Step 5: Commit**

```bash
git add backend/app/pipeline/ocr.py backend/tests/test_ocr.py
git commit -m "feat: OCR axis tick labels and titles via Tesseract"
```

---

## Task 5: Legend detection (series name + color)

**Files:**
- Create: `backend/app/pipeline/legend.py`
- Test: `backend/tests/test_legend.py`

- [ ] **Step 1: Write the failing test**

`backend/tests/test_legend.py`:
```python
from app.pipeline.geometry import detect_plot_box
from app.pipeline.legend import detect_legend_entries
from tests.fixtures import make_km_chart


def test_detect_legend_entries_finds_both_arms():
    image, truth = make_km_chart()
    box = detect_plot_box(image)

    entries = detect_legend_entries(image, box)

    assert len(entries) == 2
    names = {entry.name.strip().lower() for entry in entries}
    assert any("arm a" in name for name in names)
    assert any("arm b" in name for name in names)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/test_legend.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.pipeline.legend'`

- [ ] **Step 3: Implement legend detection**

`backend/app/pipeline/legend.py`:
```python
"""Finds legend entries by looking for small solid-color swatches followed
immediately (to the right) by a run of text, then OCRs that text. Legends in
matplotlib-style figures are drawn as a tight box of swatch+label rows, so
scanning for that pattern generalizes reasonably well beyond matplotlib too."""
from dataclasses import dataclass

import cv2
import numpy as np
import pytesseract

from app.pipeline.geometry import PlotBox

_MIN_SWATCH_AREA = 40
_MAX_SWATCH_AREA = 2000
_SWATCH_ASPECT_RANGE = (0.3, 3.0)  # roughly square-ish to wide rectangle


@dataclass
class LegendEntry:
    name: str
    color_bgr: tuple[int, int, int]
    swatch_position: tuple[int, int]  # (x, y) center, for matching to curve colors


def _is_near_grayscale(color_bgr: tuple[int, int, int], tolerance: int = 12) -> bool:
    b, g, r = color_bgr
    return max(b, g, r) - min(b, g, r) < tolerance


def detect_legend_entries(image: np.ndarray, box: PlotBox) -> list[LegendEntry]:
    # Search inside the plot box for small saturated-color rectangular blobs
    # (candidate legend swatches) — legends are typically drawn inside or
    # just outside the axes.
    region = image[box.top:box.bottom, box.left:box.right]
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    saturated_mask = cv2.inRange(hsv, (0, 80, 60), (179, 255, 255))

    contours, _ = cv2.findContours(saturated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    entries: list[LegendEntry] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if not (_MIN_SWATCH_AREA <= area <= _MAX_SWATCH_AREA):
            continue
        aspect = w / h if h else 0
        if not (_SWATCH_ASPECT_RANGE[0] <= aspect <= _SWATCH_ASPECT_RANGE[1]):
            continue

        swatch_color = region[y + h // 2, x + w // 2].tolist()
        if _is_near_grayscale(tuple(swatch_color)):
            continue  # likely text or axis artifact, not a color swatch

        text_crop = region[max(0, y - 4): y + h + 4, x + w + 3: min(region.shape[1], x + w + 150)]
        if text_crop.size == 0:
            continue

        gray = cv2.cvtColor(text_crop, cv2.COLOR_BGR2GRAY)
        upscaled = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        _, binary = cv2.threshold(upscaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        text = pytesseract.image_to_string(binary, config="--psm 7").strip()

        if not text:
            continue

        entries.append(
            LegendEntry(
                name=text,
                color_bgr=tuple(swatch_color),
                swatch_position=(box.left + x + w // 2, box.top + y + h // 2),
            )
        )

    return entries
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && pytest tests/test_legend.py -v`
Expected: PASS. If swatches aren't found, print `saturated_mask` contour count and loosen `_MIN_SWATCH_AREA`/`_SWATCH_ASPECT_RANGE`; if OCR text is empty, widen the `text_crop` width.

- [ ] **Step 5: Commit**

```bash
git add backend/app/pipeline/legend.py backend/tests/test_legend.py
git commit -m "feat: detect legend entries (color swatch + series name)"
```

---

## Task 6: Pixel-to-data calibration (linear + log)

**Files:**
- Create: `backend/app/pipeline/calibration.py`
- Test: `backend/tests/test_calibration.py`

- [ ] **Step 1: Write the failing test**

`backend/tests/test_calibration.py`:
```python
import pytest

from app.pipeline.calibration import AxisCalibration, fit_axis_calibration
from app.pipeline.ocr import TickLabel


def test_fit_linear_calibration_maps_pixels_to_values():
    # 5 ticks evenly spaced in pixels, matching evenly spaced values 0..100
    labels = [
        TickLabel(pixel_position=300, text="0", value=0.0),
        TickLabel(pixel_position=225, text="25", value=25.0),
        TickLabel(pixel_position=150, text="50", value=50.0),
        TickLabel(pixel_position=75, text="75", value=75.0),
        TickLabel(pixel_position=0, text="100", value=100.0),
    ]

    calibration = fit_axis_calibration(labels, log_scale=False)

    assert calibration.pixel_to_value(300) == pytest.approx(0.0, abs=0.5)
    assert calibration.pixel_to_value(150) == pytest.approx(50.0, abs=0.5)
    assert calibration.pixel_to_value(0) == pytest.approx(100.0, abs=0.5)


def test_fit_calibration_ignores_unreadable_ticks():
    labels = [
        TickLabel(pixel_position=300, text="0", value=0.0),
        TickLabel(pixel_position=225, text="", value=None),  # OCR failed on this one
        TickLabel(pixel_position=150, text="50", value=50.0),
        TickLabel(pixel_position=0, text="100", value=100.0),
    ]

    calibration = fit_axis_calibration(labels, log_scale=False)

    assert calibration.pixel_to_value(150) == pytest.approx(50.0, abs=0.5)


def test_fit_calibration_raises_with_fewer_than_two_readable_ticks():
    labels = [TickLabel(pixel_position=300, text="0", value=0.0)]

    with pytest.raises(ValueError, match="at least 2"):
        fit_axis_calibration(labels, log_scale=False)


def test_fit_log_calibration():
    labels = [
        TickLabel(pixel_position=300, text="1", value=1.0),
        TickLabel(pixel_position=200, text="10", value=10.0),
        TickLabel(pixel_position=100, text="100", value=100.0),
    ]

    calibration = fit_axis_calibration(labels, log_scale=True)

    assert calibration.pixel_to_value(300) == pytest.approx(1.0, rel=0.05)
    assert calibration.pixel_to_value(200) == pytest.approx(10.0, rel=0.05)
    assert calibration.pixel_to_value(100) == pytest.approx(100.0, rel=0.05)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/test_calibration.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.pipeline.calibration'`

- [ ] **Step 3: Implement calibration fitting**

`backend/app/pipeline/calibration.py`:
```python
"""Fits a pixel-position -> data-value mapping from a set of OCR'd tick
labels, using least-squares linear regression (or log10-space regression for
log-scaled axes). Ticks whose OCR text didn't parse as a number are dropped
before fitting."""
from dataclasses import dataclass

import numpy as np

from app.pipeline.ocr import TickLabel


@dataclass
class AxisCalibration:
    slope: float
    intercept: float
    log_scale: bool

    def pixel_to_value(self, pixel: float) -> float:
        raw = self.slope * pixel + self.intercept
        return 10 ** raw if self.log_scale else raw


def fit_axis_calibration(labels: list[TickLabel], log_scale: bool) -> AxisCalibration:
    readable = [label for label in labels if label.value is not None]
    if len(readable) < 2:
        raise ValueError(f"Need at least 2 readable tick labels to calibrate, got {len(readable)}")

    pixels = np.array([label.pixel_position for label in readable], dtype=float)
    values = np.array([label.value for label in readable], dtype=float)

    if log_scale:
        if np.any(values <= 0):
            raise ValueError("Log-scale axis cannot have zero or negative tick values")
        values = np.log10(values)

    slope, intercept = np.polyfit(pixels, values, deg=1)
    return AxisCalibration(slope=float(slope), intercept=float(intercept), log_scale=log_scale)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && pytest tests/test_calibration.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/pipeline/calibration.py backend/tests/test_calibration.py
git commit -m "feat: fit pixel-to-data axis calibration (linear and log scale)"
```

---

## Task 7: Chart-type classification heuristic

**Files:**
- Create: `backend/app/pipeline/classify.py`
- Test: `backend/tests/test_classify.py`

- [ ] **Step 1: Write the failing test**

`backend/tests/test_classify.py`:
```python
from app.pipeline.classify import ChartType, classify_chart_type
from app.pipeline.geometry import detect_plot_box
from tests.fixtures import make_bar_chart, make_km_chart, make_line_chart, make_scatter_chart


def test_classify_line_chart():
    image, _ = make_line_chart()
    box = detect_plot_box(image)
    assert classify_chart_type(image, box) == ChartType.LINE


def test_classify_scatter_chart():
    image, _ = make_scatter_chart()
    box = detect_plot_box(image)
    assert classify_chart_type(image, box) == ChartType.SCATTER


def test_classify_bar_chart():
    image, _ = make_bar_chart()
    box = detect_plot_box(image)
    assert classify_chart_type(image, box) == ChartType.BAR


def test_classify_km_chart():
    image, _ = make_km_chart()
    box = detect_plot_box(image)
    assert classify_chart_type(image, box) == ChartType.KAPLAN_MEIER
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/test_classify.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.pipeline.classify'`

- [ ] **Step 3: Implement classification heuristic**

`backend/app/pipeline/classify.py`:
```python
"""Classifies chart type from plotted shape characteristics — no OCR or
labels needed. Order of checks matters: bar charts are ruled in first since
they have the most distinctive signature (large filled rectangular blobs
sharing a common baseline), then step-function detection distinguishes
Kaplan-Meier curves from ordinary line charts, then remaining charts are
split into line vs. scatter by how much of each column-of-pixels forms a
continuous stroke versus isolated blobs."""
from enum import Enum

import cv2
import numpy as np

from app.pipeline.geometry import PlotBox


class ChartType(str, Enum):
    LINE = "line"
    SCATTER = "scatter"
    BAR = "bar"
    KAPLAN_MEIER = "kaplan_meier"


def _non_background_mask(region: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)
    return mask


def _is_bar_chart(mask: np.ndarray) -> bool:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h = mask.shape[0]
    bar_like = 0
    for contour in contours:
        x, y, w, cy = cv2.boundingRect(contour)
        area = w * cy
        if area < mask.size * 0.01:
            continue
        # bars are tall filled rectangles that touch (or nearly touch) the bottom
        fill_ratio = cv2.contourArea(contour) / area if area else 0
        touches_bottom = (y + cy) >= h - 5
        if fill_ratio > 0.85 and touches_bottom and cy > w:
            bar_like += 1
    return bar_like >= 1


def _has_step_pattern(mask: np.ndarray) -> bool:
    """Step functions have many horizontal runs connected by short vertical
    jumps — approximate by counting columns where the stroke's vertical
    extent is near-zero (flat run) versus columns with a tall vertical jump."""
    column_extents = []
    for col in range(mask.shape[1]):
        rows = np.where(mask[:, col] > 0)[0]
        if len(rows) == 0:
            continue
        column_extents.append(rows.max() - rows.min())

    if not column_extents:
        return False

    extents = np.array(column_extents)
    flat_columns = np.sum(extents < 3)
    jump_columns = np.sum(extents > mask.shape[0] * 0.05)
    return flat_columns > len(extents) * 0.5 and jump_columns > 0


def _is_scatter(mask: np.ndarray) -> bool:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    small_blobs = [c for c in contours if cv2.contourArea(c) < mask.size * 0.02]
    # scatter: many small, disconnected blobs rather than one continuous stroke
    return len(small_blobs) >= 6


def classify_chart_type(image: np.ndarray, box: PlotBox) -> ChartType:
    region = image[box.top:box.bottom, box.left:box.right]
    mask = _non_background_mask(region)

    if _is_bar_chart(mask):
        return ChartType.BAR
    if _has_step_pattern(mask):
        return ChartType.KAPLAN_MEIER
    if _is_scatter(mask):
        return ChartType.SCATTER
    return ChartType.LINE
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && pytest tests/test_classify.py -v`
Expected: PASS. If a chart type is misclassified, print the intermediate `mask` stats (`flat_columns`, `jump_columns`, blob count) for that fixture and tune the thresholds — these are heuristics, not exact rules, and thresholds may need adjusting once tested against real (non-synthetic) chart images later in Task 15.

- [ ] **Step 5: Commit**

```bash
git add backend/app/pipeline/classify.py backend/tests/test_classify.py
git commit -m "feat: classify chart type (line/scatter/bar/Kaplan-Meier) from shape heuristics"
```

---

## Task 8: Series color-mask isolation

Every extraction routine (line tracing, scatter centroids, bar extents) needs to first isolate "just this series' pixels" from the plot area. Build that once as a shared building block.

**Files:**
- Create: `backend/app/pipeline/curves.py`
- Test: `backend/tests/test_curves_mask.py`

- [ ] **Step 1: Write the failing test**

`backend/tests/test_curves_mask.py`:
```python
import numpy as np

from app.pipeline.curves import isolate_series_mask
from app.pipeline.geometry import detect_plot_box
from tests.fixtures import make_km_chart


def test_isolate_series_mask_separates_red_and_green_arms():
    image, _ = make_km_chart()
    box = detect_plot_box(image)

    red_mask = isolate_series_mask(image, box, target_color_bgr=(0, 0, 255), tolerance=40)
    green_mask = isolate_series_mask(image, box, target_color_bgr=(0, 255, 0), tolerance=40)

    assert red_mask.sum() > 0
    assert green_mask.sum() > 0
    # the two masks should be almost entirely disjoint
    overlap = np.logical_and(red_mask > 0, green_mask > 0).sum()
    assert overlap < min(red_mask.sum(), green_mask.sum()) * 0.05 / 255
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/test_curves_mask.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.pipeline.curves'`

- [ ] **Step 3: Implement color-mask isolation**

`backend/app/pipeline/curves.py`:
```python
"""Isolates and traces individual series' pixels from the plot area, and
extracts (x, y) data points for line/step, scatter, and bar chart types."""
import cv2
import numpy as np

from app.pipeline.calibration import AxisCalibration
from app.pipeline.geometry import PlotBox


def isolate_series_mask(
    image: np.ndarray, box: PlotBox, target_color_bgr: tuple[int, int, int], tolerance: int = 40
) -> np.ndarray:
    region = image[box.top:box.bottom, box.left:box.right]
    target = np.array(target_color_bgr, dtype=np.int16)
    diff = np.abs(region.astype(np.int16) - target).sum(axis=2)
    mask = np.where(diff <= tolerance, 255, 0).astype(np.uint8)
    return mask
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && pytest tests/test_curves_mask.py -v`
Expected: PASS. If overlap is too high, lower `tolerance`; if a mask sum is 0, the fixture's line color didn't match `target_color_bgr` closely enough — print `region[region.shape[0]//2]` to sample actual pixel colors along the curve.

- [ ] **Step 5: Commit**

```bash
git add backend/app/pipeline/curves.py backend/tests/test_curves_mask.py
git commit -m "feat: isolate per-series pixel mask by target color"
```

---

## Task 9: Line and Kaplan-Meier step-curve tracing

**Files:**
- Modify: `backend/app/pipeline/curves.py`
- Test: `backend/tests/test_curves_line.py`

- [ ] **Step 1: Write the failing test**

`backend/tests/test_curves_line.py`:
```python
import pytest

from app.pipeline.calibration import fit_axis_calibration
from app.pipeline.curves import isolate_series_mask, trace_line_curve
from app.pipeline.geometry import detect_plot_box, detect_tick_positions
from app.pipeline.ocr import read_axis_tick_labels
from tests.fixtures import make_km_chart, make_line_chart


def _calibrate(image, box):
    x_ticks = detect_tick_positions(image, box, axis="x")
    y_ticks = detect_tick_positions(image, box, axis="y")
    x_labels = read_axis_tick_labels(image, box, x_ticks, axis="x")
    y_labels = read_axis_tick_labels(image, box, y_ticks, axis="y")
    return (
        fit_axis_calibration(x_labels, log_scale=False),
        fit_axis_calibration(y_labels, log_scale=False),
    )


def test_trace_line_curve_recovers_known_line():
    image, truth = make_line_chart()
    box = detect_plot_box(image)
    x_cal, y_cal = _calibrate(image, box)

    mask = isolate_series_mask(image, box, target_color_bgr=(255, 0, 0), tolerance=60)  # blue
    points = trace_line_curve(mask, box, x_cal, y_cal, step=False)

    assert len(points) > 5
    # y = 8x + 5 in the fixture; check a mid-range traced point is close
    mid_points = [p for p in points if 4 <= p[0] <= 6]
    assert mid_points
    x, y = mid_points[len(mid_points) // 2]
    assert y == pytest.approx(8 * x + 5, abs=5)


def test_trace_line_curve_step_mode_recovers_km_curve():
    image, truth = make_km_chart()
    box = detect_plot_box(image)
    x_cal, y_cal = _calibrate(image, box)

    mask = isolate_series_mask(image, box, target_color_bgr=(0, 0, 255), tolerance=60)  # red
    points = trace_line_curve(mask, box, x_cal, y_cal, step=True)

    assert len(points) > 3
    # survival should be non-increasing across a KM curve
    y_values = [p[1] for p in points]
    assert all(y_values[i] >= y_values[i + 1] - 0.05 for i in range(len(y_values) - 1))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/test_curves_line.py -v`
Expected: FAIL — `ImportError: cannot import name 'trace_line_curve'`

- [ ] **Step 3: Implement line/step tracing**

Append to `backend/app/pipeline/curves.py`:
```python
def trace_line_curve(
    mask: np.ndarray,
    box: PlotBox,
    x_calibration: AxisCalibration,
    y_calibration: AxisCalibration,
    step: bool,
) -> list[tuple[float, float]]:
    """For each pixel-column containing series pixels, takes the vertical
    center (line mode) or, for step-function curves, keeps every distinct
    flat run's y-level so vertical jumps are preserved as separate points."""
    points_px: list[tuple[int, int]] = []

    for col in range(mask.shape[1]):
        rows = np.where(mask[:, col] > 0)[0]
        if len(rows) == 0:
            continue
        if step:
            # a step curve can have two y-levels in the same column at a
            # jump; record the extremes rather than averaging them away
            points_px.append((col, int(rows.min())))
            if rows.max() != rows.min():
                points_px.append((col, int(rows.max())))
        else:
            points_px.append((col, int(rows.mean())))

    points_px.sort(key=lambda p: p[0])

    points_data = [
        (
            x_calibration.pixel_to_value(box.left + col),
            y_calibration.pixel_to_value(box.top + row),
        )
        for col, row in points_px
    ]
    return points_data
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && pytest tests/test_curves_line.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/pipeline/curves.py backend/tests/test_curves_line.py
git commit -m "feat: trace line and step-function (Kaplan-Meier) curves to data points"
```

---

## Task 10: Scatter centroid and bar extent extraction

**Files:**
- Modify: `backend/app/pipeline/curves.py`
- Test: `backend/tests/test_curves_scatter_bar.py`

- [ ] **Step 1: Write the failing test**

`backend/tests/test_curves_scatter_bar.py`:
```python
import pytest

from app.pipeline.calibration import fit_axis_calibration
from app.pipeline.curves import extract_bar_heights, extract_scatter_points, isolate_series_mask
from app.pipeline.geometry import detect_plot_box, detect_tick_positions
from app.pipeline.ocr import read_axis_tick_labels
from tests.fixtures import make_bar_chart, make_scatter_chart


def _calibrate(image, box):
    x_ticks = detect_tick_positions(image, box, axis="x")
    y_ticks = detect_tick_positions(image, box, axis="y")
    x_labels = read_axis_tick_labels(image, box, x_ticks, axis="x")
    y_labels = read_axis_tick_labels(image, box, y_ticks, axis="y")
    return (
        fit_axis_calibration(x_labels, log_scale=False),
        fit_axis_calibration(y_labels, log_scale=False),
    )


def test_extract_scatter_points_recovers_approximate_count():
    image, truth = make_scatter_chart()
    box = detect_plot_box(image)
    x_cal, y_cal = _calibrate(image, box)

    mask = isolate_series_mask(image, box, target_color_bgr=(128, 0, 128), tolerance=70)  # purple
    points = extract_scatter_points(mask, box, x_cal, y_cal)

    assert len(points) == pytest.approx(len(truth["series"][0]["points"]), abs=3)


def test_extract_bar_heights_recovers_known_heights():
    image, truth = make_bar_chart()
    box = detect_plot_box(image)
    x_cal, y_cal = _calibrate(image, box)

    mask = isolate_series_mask(image, box, target_color_bgr=(128, 128, 0), tolerance=70)  # teal
    bars = extract_bar_heights(mask, box, x_cal, y_cal)

    assert len(bars) == 4
    heights = sorted(y for _, y in bars)
    expected = sorted(y for _, y in truth["series"][0]["points"])
    for actual, exp in zip(heights, expected):
        assert actual == pytest.approx(exp, abs=3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/test_curves_scatter_bar.py -v`
Expected: FAIL — `ImportError: cannot import name 'extract_scatter_points'`

- [ ] **Step 3: Implement scatter and bar extraction**

Append to `backend/app/pipeline/curves.py`:
```python
def extract_scatter_points(
    mask: np.ndarray, box: PlotBox, x_calibration: AxisCalibration, y_calibration: AxisCalibration
) -> list[tuple[float, float]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    points_data = []
    for contour in contours:
        if cv2.contourArea(contour) < 3:
            continue
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
        points_data.append(
            (
                x_calibration.pixel_to_value(box.left + cx),
                y_calibration.pixel_to_value(box.top + cy),
            )
        )
    return points_data


def extract_bar_heights(
    mask: np.ndarray, box: PlotBox, x_calibration: AxisCalibration, y_calibration: AxisCalibration
) -> list[tuple[float, float]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bars = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h < mask.size * 0.005:
            continue
        center_x_px = box.left + x + w / 2
        top_y_px = box.top + y  # top edge of the bar = its value
        bars.append(
            (
                x_calibration.pixel_to_value(center_x_px),
                y_calibration.pixel_to_value(top_y_px),
            )
        )
    bars.sort(key=lambda p: p[0])
    return bars
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && pytest tests/test_curves_scatter_bar.py -v`
Expected: PASS. If bar count is wrong, adjacent bars may be merging in the mask — check the fixture's bar spacing vs. `tolerance` used in `isolate_series_mask`.

- [ ] **Step 5: Commit**

```bash
git add backend/app/pipeline/curves.py backend/tests/test_curves_scatter_bar.py
git commit -m "feat: extract scatter point centroids and bar heights"
```

---

## Task 11: Kaplan-Meier censoring tick detection

**Files:**
- Modify: `backend/app/pipeline/curves.py`
- Test: `backend/tests/test_curves_censoring.py`

- [ ] **Step 1: Write the failing test**

`backend/tests/test_curves_censoring.py`:
```python
from app.pipeline.calibration import fit_axis_calibration
from app.pipeline.curves import detect_censoring_marks, isolate_series_mask
from app.pipeline.geometry import detect_plot_box, detect_tick_positions
from app.pipeline.ocr import read_axis_tick_labels
from tests.fixtures import make_km_chart


def test_detect_censoring_marks_finds_plus_markers():
    image, truth = make_km_chart()
    box = detect_plot_box(image)
    x_ticks = detect_tick_positions(image, box, axis="x")
    y_ticks = detect_tick_positions(image, box, axis="y")
    x_cal = fit_axis_calibration(read_axis_tick_labels(image, box, x_ticks, axis="x"), log_scale=False)
    y_cal = fit_axis_calibration(read_axis_tick_labels(image, box, y_ticks, axis="y"), log_scale=False)

    mask = isolate_series_mask(image, box, target_color_bgr=(0, 0, 255), tolerance=60)  # red arm
    marks = detect_censoring_marks(mask, box, x_cal, y_cal)

    # fixture has 2 censoring points on the red arm, at t=5 and t=9
    assert len(marks) == 2
    times = sorted(x for x, _ in marks)
    assert times[0] == pytest.approx(5, abs=1)
    assert times[1] == pytest.approx(9, abs=1)
```

Add `import pytest` at the top of the test file alongside the other imports.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/test_curves_censoring.py -v`
Expected: FAIL — `ImportError: cannot import name 'detect_censoring_marks'`

- [ ] **Step 3: Implement censoring mark detection**

Append to `backend/app/pipeline/curves.py`:
```python
def detect_censoring_marks(
    mask: np.ndarray, box: PlotBox, x_calibration: AxisCalibration, y_calibration: AxisCalibration
) -> list[tuple[float, float]]:
    """Censoring ticks (typically '+' or '|' markers) are small isolated
    blobs disconnected from the main step-curve stroke, roughly
    square/plus-shaped rather than long thin horizontal/vertical runs. This
    finds small blobs whose bounding box is close to square and clearly
    smaller than a full step-run segment."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    marks = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area < 6 or area > 400:
            continue
        aspect = w / h if h else 0
        if not (0.4 <= aspect <= 2.5):
            continue  # skip long thin strokes (part of the main curve line)

        center_x_px = box.left + x + w / 2
        center_y_px = box.top + y + h / 2
        marks.append(
            (
                x_calibration.pixel_to_value(center_x_px),
                y_calibration.pixel_to_value(center_y_px),
            )
        )
    return marks
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && pytest tests/test_curves_censoring.py -v`
Expected: PASS. If marks are missed or the main curve stroke is picked up as a false mark, tune the `area`/`aspect` thresholds against this fixture's actual marker pixel size.

- [ ] **Step 5: Commit**

```bash
git add backend/app/pipeline/curves.py backend/tests/test_curves_censoring.py
git commit -m "feat: detect Kaplan-Meier censoring tick marks"
```

---

## Task 12: Overlay image generation

**Files:**
- Create: `backend/app/pipeline/overlay.py`
- Test: `backend/tests/test_overlay.py`

- [ ] **Step 1: Write the failing test**

`backend/tests/test_overlay.py`:
```python
import numpy as np

from app.pipeline.calibration import fit_axis_calibration
from app.pipeline.geometry import detect_plot_box, detect_tick_positions
from app.pipeline.ocr import read_axis_tick_labels
from app.pipeline.overlay import draw_overlay
from tests.fixtures import make_line_chart


def test_draw_overlay_returns_modified_image_same_shape():
    image, _ = make_line_chart()
    box = detect_plot_box(image)
    x_ticks = detect_tick_positions(image, box, axis="x")
    y_ticks = detect_tick_positions(image, box, axis="y")
    x_cal = fit_axis_calibration(read_axis_tick_labels(image, box, x_ticks, axis="x"), log_scale=False)
    y_cal = fit_axis_calibration(read_axis_tick_labels(image, box, y_ticks, axis="y"), log_scale=False)

    series = [{"name": "Series A", "color_bgr": (255, 0, 0), "points": [(1.0, 13.0), (5.0, 45.0), (9.0, 77.0)]}]

    overlay_image = draw_overlay(image, box, x_cal, y_cal, series)

    assert overlay_image.shape == image.shape
    # overlay must actually change some pixels (markers drawn), not be a no-op copy
    assert not np.array_equal(overlay_image, image)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/test_overlay.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.pipeline.overlay'`

- [ ] **Step 3: Implement overlay drawing**

`backend/app/pipeline/overlay.py`:
```python
"""Draws detected data points back onto a copy of the original chart image,
for the user to visually sanity-check auto-extraction accuracy."""
import cv2
import numpy as np

from app.pipeline.calibration import AxisCalibration
from app.pipeline.geometry import PlotBox

_MARKER_RADIUS = 4
_MARKER_COLOR_BGR = (0, 0, 0)  # black outline, drawn under a white fill for contrast on any series color
_MARKER_OUTLINE_COLOR_BGR = (255, 255, 255)


def _value_to_pixel(value: float, calibration: AxisCalibration, origin: int) -> int:
    # invert calibration.pixel_to_value: value = slope*pixel + intercept (or slope*pixel+intercept in log space)
    raw_value = np.log10(value) if calibration.log_scale else value
    pixel = (raw_value - calibration.intercept) / calibration.slope
    return int(round(pixel))


def draw_overlay(
    image: np.ndarray,
    box: PlotBox,
    x_calibration: AxisCalibration,
    y_calibration: AxisCalibration,
    series: list[dict],
) -> np.ndarray:
    overlay = image.copy()

    for s in series:
        for x_val, y_val in s["points"]:
            px = _value_to_pixel(x_val, x_calibration, box.left)
            py = _value_to_pixel(y_val, y_calibration, box.top)
            cv2.circle(overlay, (px, py), _MARKER_RADIUS + 1, _MARKER_OUTLINE_COLOR_BGR, -1)
            cv2.circle(overlay, (px, py), _MARKER_RADIUS, s.get("color_bgr", _MARKER_COLOR_BGR), -1)

    return overlay
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && pytest tests/test_overlay.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/pipeline/overlay.py backend/tests/test_overlay.py
git commit -m "feat: draw detected points overlay onto original chart image"
```

---

## Task 13: Pipeline orchestration

Wires every prior module together into one entry point the API layer calls. Handles the case where no legend is found (single unlabeled series) and supports a manual axis-range override for the "OCR failed" fallback path.

**Files:**
- Create: `backend/app/pipeline/pipeline.py`
- Test: `backend/tests/test_pipeline.py`

- [ ] **Step 1: Write the failing test**

`backend/tests/test_pipeline.py`:
```python
import cv2
import pytest

from app.pipeline.pipeline import run_pipeline
from tests.fixtures import make_km_chart, make_line_chart


def _encode(image):
    success, buf = cv2.imencode(".png", image)
    assert success
    return buf.tobytes()


def test_run_pipeline_on_single_series_line_chart():
    image, truth = make_line_chart()
    result = run_pipeline(_encode(image))

    assert result.chart_type == "line"
    assert len(result.series) == 1
    assert len(result.series[0].points) > 5
    mid = [p for p in result.series[0].points if 4 <= p[0] <= 6][len(result.series[0].points) // 4]
    assert mid[1] == pytest.approx(8 * mid[0] + 5, abs=8)
    assert result.overlay_image_png  # non-empty bytes


def test_run_pipeline_on_km_chart_finds_both_arms():
    image, truth = make_km_chart()
    result = run_pipeline(_encode(image))

    assert result.chart_type == "kaplan_meier"
    assert len(result.series) == 2
    names = {s.name.strip().lower() for s in result.series}
    assert any("arm a" in n for n in names)
    assert any("arm b" in n for n in names)


def test_run_pipeline_with_manual_axis_override_skips_ocr_calibration():
    image, truth = make_line_chart()
    result = run_pipeline(
        _encode(image),
        manual_x_range=(0.0, 10.0),
        manual_y_range=(0.0, 100.0),
    )

    assert len(result.series) == 1
    assert len(result.series[0].points) > 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/test_pipeline.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.pipeline.pipeline'`

- [ ] **Step 3: Implement pipeline orchestration**

`backend/app/pipeline/pipeline.py`:
```python
"""Top-level orchestration: image bytes in, structured extraction result out.
Ties together geometry detection, OCR-based calibration (or a manual
override), chart-type classification, and per-series point extraction."""
from dataclasses import dataclass, field

import cv2
import numpy as np

from app.pipeline.calibration import AxisCalibration, fit_axis_calibration
from app.pipeline.classify import ChartType, classify_chart_type
from app.pipeline.curves import (
    detect_censoring_marks,
    extract_bar_heights,
    extract_scatter_points,
    isolate_series_mask,
    trace_line_curve,
)
from app.pipeline.geometry import PlotBox, detect_plot_box, detect_tick_positions
from app.pipeline.legend import LegendEntry, detect_legend_entries
from app.pipeline.ocr import read_axis_tick_labels


@dataclass
class SeriesResult:
    name: str
    color_bgr: tuple[int, int, int]
    points: list[tuple[float, float]]
    censoring_marks: list[tuple[float, float]] = field(default_factory=list)


@dataclass
class PipelineResult:
    chart_type: str
    series: list[SeriesResult]
    overlay_image_png: bytes
    x_axis_calibrated_from_ocr: bool
    y_axis_calibrated_from_ocr: bool
    x_reference_points: list[tuple[float, float]]  # (pixel, value) pairs used to fit x calibration
    y_reference_points: list[tuple[float, float]]  # (pixel, value) pairs used to fit y calibration


def _decode_image(image_bytes: bytes) -> np.ndarray:
    file_bytes = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image — unsupported or corrupt file")
    return image


def _build_axis_calibration(
    image: np.ndarray, box: PlotBox, axis: str, manual_range: tuple[float, float] | None
) -> tuple[AxisCalibration, bool, list[tuple[float, float]]]:
    if manual_range is not None:
        low, high = manual_range
        # two-point calibration directly from user-provided min/max at the
        # known plot-box edges, bypassing OCR entirely
        if axis == "x":
            slope = (high - low) / (box.right - box.left)
            intercept = low - slope * box.left
            reference_points = [(float(box.left), low), (float(box.right), high)]
        else:
            slope = (low - high) / (box.bottom - box.top)
            intercept = high - slope * box.top
            reference_points = [(float(box.bottom), low), (float(box.top), high)]
        return AxisCalibration(slope=slope, intercept=intercept, log_scale=False), False, reference_points

    ticks = detect_tick_positions(image, box, axis=axis)
    labels = read_axis_tick_labels(image, box, ticks, axis=axis)
    calibration = fit_axis_calibration(labels, log_scale=False)
    reference_points = [
        (float(label.pixel_position), label.value) for label in labels if label.value is not None
    ]
    return calibration, True, reference_points


def _find_series_color_for_legend(legend_entries: list[LegendEntry]) -> list[tuple[str, tuple[int, int, int]]]:
    return [(entry.name, entry.color_bgr) for entry in legend_entries]


def _dominant_series_color(image: np.ndarray, box: PlotBox) -> tuple[int, int, int]:
    """Fallback when no legend is found: assumes a single series and picks
    the most common non-white, non-grayscale color in the plot area."""
    region = image[box.top:box.bottom, box.left:box.right].reshape(-1, 3)
    is_colorful = (region.max(axis=1).astype(int) - region.min(axis=1).astype(int)) > 20
    colorful_pixels = region[is_colorful]
    if len(colorful_pixels) == 0:
        raise ValueError("No colored curve/marker pixels found in plot area")
    colors, counts = np.unique(colorful_pixels, axis=0, return_counts=True)
    return tuple(int(c) for c in colors[np.argmax(counts)])


def run_pipeline(
    image_bytes: bytes,
    manual_x_range: tuple[float, float] | None = None,
    manual_y_range: tuple[float, float] | None = None,
) -> PipelineResult:
    image = _decode_image(image_bytes)
    box = detect_plot_box(image)
    chart_type = classify_chart_type(image, box)

    x_calibration, x_from_ocr, x_reference_points = _build_axis_calibration(image, box, "x", manual_x_range)
    y_calibration, y_from_ocr, y_reference_points = _build_axis_calibration(image, box, "y", manual_y_range)

    legend_entries = detect_legend_entries(image, box)
    named_colors = _find_series_color_for_legend(legend_entries)
    if not named_colors:
        named_colors = [("Series A", _dominant_series_color(image, box))]

    series_results: list[SeriesResult] = []
    for name, color_bgr in named_colors:
        mask = isolate_series_mask(image, box, color_bgr, tolerance=60)

        if chart_type == ChartType.BAR:
            points = extract_bar_heights(mask, box, x_calibration, y_calibration)
            censoring: list[tuple[float, float]] = []
        elif chart_type == ChartType.SCATTER:
            points = extract_scatter_points(mask, box, x_calibration, y_calibration)
            censoring = []
        elif chart_type == ChartType.KAPLAN_MEIER:
            points = trace_line_curve(mask, box, x_calibration, y_calibration, step=True)
            censoring = detect_censoring_marks(mask, box, x_calibration, y_calibration)
        else:  # LINE
            points = trace_line_curve(mask, box, x_calibration, y_calibration, step=False)
            censoring = []

        series_results.append(
            SeriesResult(name=name, color_bgr=color_bgr, points=points, censoring_marks=censoring)
        )

    from app.pipeline.overlay import draw_overlay

    overlay_image = draw_overlay(
        image, box, x_calibration, y_calibration,
        [{"name": s.name, "color_bgr": s.color_bgr, "points": s.points} for s in series_results],
    )
    success, overlay_encoded = cv2.imencode(".png", overlay_image)
    if not success:
        raise ValueError("Failed to encode overlay image")

    return PipelineResult(
        chart_type=chart_type.value,
        series=series_results,
        overlay_image_png=overlay_encoded.tobytes(),
        x_axis_calibrated_from_ocr=x_from_ocr,
        y_axis_calibrated_from_ocr=y_from_ocr,
        x_reference_points=x_reference_points,
        y_reference_points=y_reference_points,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && pytest tests/test_pipeline.py -v`
Expected: PASS. This test exercises the full stack, so a failure could originate in any earlier module — check which assertion fails first and re-run that module's own test file to narrow it down.

- [ ] **Step 5: Run the full backend test suite to confirm no regressions**

Run: `cd backend && pytest -v`
Expected: all tests across every module PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/app/pipeline/pipeline.py backend/tests/test_pipeline.py
git commit -m "feat: orchestrate full extraction pipeline with manual axis override support"
```

---

## Task 14: Password auth with signed session cookie

**Files:**
- Create: `backend/app/auth.py`
- Test: `backend/tests/test_auth.py`

- [ ] **Step 1: Write the failing test**

`backend/tests/test_auth.py`:
```python
import os

os.environ["APP_PASSWORD"] = "test-password-123"
os.environ["SESSION_SECRET_KEY"] = "test-secret-key"

from app.auth import create_session_token, verify_password, verify_session_token


def test_verify_password_accepts_correct_password():
    assert verify_password("test-password-123") is True


def test_verify_password_rejects_wrong_password():
    assert verify_password("wrong") is False


def test_session_token_round_trips():
    token = create_session_token()
    assert verify_session_token(token) is True


def test_session_token_rejects_tampered_value():
    token = create_session_token()
    tampered = token[:-1] + ("a" if token[-1] != "a" else "b")
    assert verify_session_token(tampered) is False


def test_session_token_rejects_garbage():
    assert verify_session_token("not-a-real-token") is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/test_auth.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.auth'`

- [ ] **Step 3: Implement auth**

`backend/app/auth.py`:
```python
"""Single shared-password auth: on successful login, issues a signed,
timestamped session token (no server-side session storage needed since the
app is stateless). The token is opaque to the client and expires after 7
days."""
import os

from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

_SESSION_MAX_AGE_SECONDS = 7 * 24 * 60 * 60
_SESSION_PAYLOAD = "authenticated"


def _serializer() -> URLSafeTimedSerializer:
    secret_key = os.environ["SESSION_SECRET_KEY"]
    return URLSafeTimedSerializer(secret_key)


def verify_password(password: str) -> bool:
    return password == os.environ["APP_PASSWORD"]


def create_session_token() -> str:
    return _serializer().dumps(_SESSION_PAYLOAD)


def verify_session_token(token: str) -> bool:
    try:
        payload = _serializer().loads(token, max_age=_SESSION_MAX_AGE_SECONDS)
    except (BadSignature, SignatureExpired):
        return False
    return payload == _SESSION_PAYLOAD
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && pytest tests/test_auth.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/auth.py backend/tests/test_auth.py
git commit -m "feat: add password auth with signed session token"
```

---

## Task 15: Login and upload API endpoints

**Files:**
- Modify: `backend/app/main.py`
- Test: `backend/tests/test_api.py`

- [ ] **Step 1: Write the failing test**

`backend/tests/test_api.py`:
```python
import os

os.environ["APP_PASSWORD"] = "test-password-123"
os.environ["SESSION_SECRET_KEY"] = "test-secret-key"

import cv2
from fastapi.testclient import TestClient

from app.main import app
from tests.fixtures import make_line_chart

client = TestClient(app)


def _login_client():
    response = client.post("/api/login", json={"password": "test-password-123"})
    assert response.status_code == 200
    assert "session" in response.cookies
    return response.cookies


def test_login_with_correct_password_sets_session_cookie():
    _login_client()


def test_login_with_wrong_password_returns_401():
    response = client.post("/api/login", json={"password": "wrong"})
    assert response.status_code == 401


def test_upload_without_session_returns_401():
    image, _ = make_line_chart()
    success, buf = cv2.imencode(".png", image)
    response = client.post("/api/upload", files={"file": ("chart.png", buf.tobytes(), "image/png")})
    assert response.status_code == 401


def test_upload_with_session_returns_extracted_series():
    cookies = _login_client()
    image, _ = make_line_chart()
    success, buf = cv2.imencode(".png", image)

    response = client.post(
        "/api/upload",
        files={"file": ("chart.png", buf.tobytes(), "image/png")},
        cookies=cookies,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["chart_type"] == "line"
    assert len(body["series"]) == 1
    assert len(body["series"][0]["points"]) > 5
    assert "overlay_image_base64" in body
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/test_api.py -v`
Expected: FAIL — 404s, since `/api/login` and `/api/upload` don't exist yet.

- [ ] **Step 3: Implement the endpoints**

`backend/app/models.py`:
```python
from pydantic import BaseModel


class LoginRequest(BaseModel):
    password: str


class SeriesResponse(BaseModel):
    name: str
    color_bgr: tuple[int, int, int]
    points: list[tuple[float, float]]
    censoring_marks: list[tuple[float, float]]


class UploadResponse(BaseModel):
    chart_type: str
    series: list[SeriesResponse]
    overlay_image_base64: str
    x_axis_calibrated_from_ocr: bool
    y_axis_calibrated_from_ocr: bool
    x_reference_points: list[tuple[float, float]]
    y_reference_points: list[tuple[float, float]]
```

Replace `backend/app/main.py` with:
```python
import base64

from fastapi import Cookie, FastAPI, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.auth import create_session_token, verify_password, verify_session_token
from app.models import LoginRequest, SeriesResponse, UploadResponse
from app.pipeline.pipeline import run_pipeline

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


@app.post("/api/login")
def login(request: LoginRequest, response: Response):
    if not verify_password(request.password):
        raise HTTPException(status_code=401, detail="Incorrect password")

    token = create_session_token()
    response.set_cookie("session", token, httponly=True, samesite="none", secure=True, max_age=7 * 24 * 60 * 60)
    return {"status": "ok"}


def _require_session(session: str | None) -> None:
    if session is None or not verify_session_token(session):
        raise HTTPException(status_code=401, detail="Not authenticated")


@app.post("/api/upload", response_model=UploadResponse)
async def upload(file: UploadFile, session: str | None = Cookie(default=None)):
    _require_session(session)

    image_bytes = await file.read()
    try:
        result = run_pipeline(image_bytes)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error))

    return UploadResponse(
        chart_type=result.chart_type,
        series=[
            SeriesResponse(
                name=s.name, color_bgr=s.color_bgr, points=s.points, censoring_marks=s.censoring_marks
            )
            for s in result.series
        ],
        overlay_image_base64=base64.b64encode(result.overlay_image_png).decode("ascii"),
        x_axis_calibrated_from_ocr=result.x_axis_calibrated_from_ocr,
        y_axis_calibrated_from_ocr=result.y_axis_calibrated_from_ocr,
        x_reference_points=result.x_reference_points,
        y_reference_points=result.y_reference_points,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && pytest tests/test_api.py -v`
Expected: PASS

- [ ] **Step 5: Run the full backend test suite**

Run: `cd backend && pytest -v`
Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/app/main.py backend/app/models.py backend/tests/test_api.py
git commit -m "feat: add login and upload API endpoints"
```

---

## Task 16: CSV/Excel export endpoint

Takes the (possibly user-corrected) series data back from the frontend and returns a downloadable file — export happens from whatever the user has on screen after edits, not by re-running the pipeline.

**Files:**
- Create: `backend/app/export.py`
- Modify: `backend/app/main.py`
- Test: `backend/tests/test_export.py`

- [ ] **Step 1: Write the failing test**

`backend/tests/test_export.py`:
```python
import io

import openpyxl

from app.export import series_to_csv, series_to_excel


def test_series_to_csv_includes_series_column_and_all_points():
    series = [
        {"name": "Arm A", "points": [(0.0, 1.0), (2.0, 0.95)]},
        {"name": "Arm B", "points": [(0.0, 1.0), (2.0, 0.90)]},
    ]

    csv_text = series_to_csv(series)

    lines = csv_text.strip().splitlines()
    assert lines[0] == "series,x,y"
    assert "Arm A,0.0,1.0" in csv_text
    assert "Arm B,2.0,0.9" in csv_text
    assert len(lines) == 5  # header + 4 data rows


def test_series_to_excel_produces_readable_workbook():
    series = [{"name": "Series A", "points": [(1.0, 13.0), (5.0, 45.0)]}]

    excel_bytes = series_to_excel(series)

    workbook = openpyxl.load_workbook(io.BytesIO(excel_bytes))
    sheet = workbook.active
    rows = list(sheet.iter_rows(values_only=True))
    assert rows[0] == ("series", "x", "y")
    assert rows[1] == ("Series A", 1.0, 13.0)
    assert rows[2] == ("Series A", 5.0, 45.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/test_export.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.export'`

- [ ] **Step 3: Implement export**

`backend/app/export.py`:
```python
"""Converts corrected series data (as sent back from the frontend after
review/editing) into downloadable CSV or Excel files."""
import io

import openpyxl


def series_to_csv(series: list[dict]) -> str:
    lines = ["series,x,y"]
    for s in series:
        for x, y in s["points"]:
            lines.append(f"{s['name']},{x},{y}")
    return "\n".join(lines) + "\n"


def series_to_excel(series: list[dict]) -> bytes:
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.append(["series", "x", "y"])
    for s in series:
        for x, y in s["points"]:
            sheet.append([s["name"], x, y])

    buffer = io.BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && pytest tests/test_export.py -v`
Expected: PASS

- [ ] **Step 5: Add the export endpoints to the API**

Add to `backend/app/models.py`:
```python
class SeriesInput(BaseModel):
    name: str
    points: list[tuple[float, float]]


class ExportRequest(BaseModel):
    series: list[SeriesInput]
```

Add to `backend/app/main.py` (below the existing imports, add `from fastapi.responses import Response as FileResponse` is unnecessary — reuse `Response` already imported; add these routes after `upload`):
```python
from app.export import series_to_csv, series_to_excel
from app.models import ExportRequest


@app.post("/api/export/csv")
def export_csv(request: ExportRequest, session: str | None = Cookie(default=None)):
    _require_session(session)
    csv_text = series_to_csv([s.model_dump() for s in request.series])
    return Response(
        content=csv_text,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=chart_data.csv"},
    )


@app.post("/api/export/excel")
def export_excel(request: ExportRequest, session: str | None = Cookie(default=None)):
    _require_session(session)
    excel_bytes = series_to_excel([s.model_dump() for s in request.series])
    return Response(
        content=excel_bytes,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=chart_data.xlsx"},
    )
```

- [ ] **Step 6: Write and run an API-level test for both export routes**

Add to `backend/tests/test_api.py`:
```python
def test_export_csv_requires_session():
    response = client.post("/api/export/csv", json={"series": []})
    assert response.status_code == 401


def test_export_csv_returns_file():
    cookies = _login_client()
    response = client.post(
        "/api/export/csv",
        json={"series": [{"name": "Series A", "points": [[1.0, 2.0]]}]},
        cookies=cookies,
    )
    assert response.status_code == 200
    assert "series,x,y" in response.text
```

Run: `cd backend && pytest tests/test_api.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add backend/app/export.py backend/app/main.py backend/app/models.py backend/tests/test_export.py backend/tests/test_api.py
git commit -m "feat: add CSV/Excel export endpoints"
```

---

## Task 17: Frontend scaffolding

**Files:**
- Create: `frontend/package.json`, `frontend/vite.config.ts`, `frontend/tsconfig.json`, `frontend/index.html`
- Create: `frontend/src/main.tsx`, `frontend/src/App.tsx`, `frontend/src/types.ts`

- [ ] **Step 1: Scaffold the Vite React-TS project**

```bash
cd frontend
npm create vite@latest . -- --template react-ts
npm install
npm install react-konva konva react-router-dom
npm install -D vitest @testing-library/react @testing-library/jest-dom jsdom
```

- [ ] **Step 2: Add a Vitest config block to `frontend/vite.config.ts`**

```typescript
/// <reference types="vitest/config" />
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  test: {
    environment: "jsdom",
    globals: true,
  },
});
```

- [ ] **Step 3: Define shared types**

`frontend/src/types.ts`:
```typescript
export interface SeriesData {
  name: string;
  colorBgr: [number, number, number];
  points: [number, number][];
  censoringMarks: [number, number][];
}

export interface UploadResult {
  chartType: string;
  series: SeriesData[];
  overlayImageBase64: string;
  xAxisCalibratedFromOcr: boolean;
  yAxisCalibratedFromOcr: boolean;
  xReferencePoints: [number, number][]; // (pixel, value) pairs
  yReferencePoints: [number, number][];
}
```

- [ ] **Step 4: Verify the scaffold builds**

Run: `cd frontend && npm run build`
Expected: build succeeds with no errors (default Vite template output).

- [ ] **Step 5: Commit**

```bash
git add frontend/
git commit -m "feat: scaffold Vite React-TypeScript frontend"
```

---

## Task 18: API client wrapper

**Files:**
- Create: `frontend/src/api.ts`
- Test: `frontend/src/api.test.ts`

- [ ] **Step 1: Write the failing test**

`frontend/src/api.test.ts`:
```typescript
import { afterEach, describe, expect, it, vi } from "vitest";
import { login, uploadChart } from "./api";

describe("api client", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("login posts the password and resolves on success", async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true });
    vi.stubGlobal("fetch", fetchMock);

    await login("hunter2");

    expect(fetchMock).toHaveBeenCalledWith(
      expect.stringContaining("/api/login"),
      expect.objectContaining({
        method: "POST",
        credentials: "include",
        body: JSON.stringify({ password: "hunter2" }),
      }),
    );
  });

  it("login throws on a 401 response", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: false, status: 401 }));

    await expect(login("wrong")).rejects.toThrow();
  });

  it("uploadChart sends the file and returns parsed JSON", async () => {
    const responseBody = {
      chart_type: "line",
      series: [],
      overlay_image_base64: "",
      x_axis_calibrated_from_ocr: true,
      y_axis_calibrated_from_ocr: true,
      x_reference_points: [[10, 0], [300, 10]],
      y_reference_points: [[10, 100], [300, 0]],
    };
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({ ok: true, json: async () => responseBody }),
    );

    const file = new File(["fake"], "chart.png", { type: "image/png" });
    const result = await uploadChart(file);

    expect(result.chartType).toBe("line");
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npx vitest run src/api.test.ts`
Expected: FAIL — `src/api.ts` doesn't exist.

- [ ] **Step 3: Implement the API client**

`frontend/src/api.ts`:
```typescript
import type { SeriesData, UploadResult } from "./types";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export async function login(password: string): Promise<void> {
  const response = await fetch(`${API_BASE}/api/login`, {
    method: "POST",
    credentials: "include",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ password }),
  });
  if (!response.ok) {
    throw new Error("Login failed");
  }
}

export async function uploadChart(file: File): Promise<UploadResult> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE}/api/upload`, {
    method: "POST",
    credentials: "include",
    body: formData,
  });
  if (!response.ok) {
    throw new Error("Upload failed");
  }
  const body = await response.json();
  return {
    chartType: body.chart_type,
    series: body.series.map((s: any) => ({
      name: s.name,
      colorBgr: s.color_bgr,
      points: s.points,
      censoringMarks: s.censoring_marks,
    })),
    overlayImageBase64: body.overlay_image_base64,
    xAxisCalibratedFromOcr: body.x_axis_calibrated_from_ocr,
    yAxisCalibratedFromOcr: body.y_axis_calibrated_from_ocr,
    xReferencePoints: body.x_reference_points,
    yReferencePoints: body.y_reference_points,
  };
}

async function downloadExport(path: string, series: SeriesData[], filename: string): Promise<void> {
  const response = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    credentials: "include",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ series: series.map((s) => ({ name: s.name, points: s.points })) }),
  });
  if (!response.ok) {
    throw new Error("Export failed");
  }
  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

export const exportCsv = (series: SeriesData[]) => downloadExport("/api/export/csv", series, "chart_data.csv");
export const exportExcel = (series: SeriesData[]) => downloadExport("/api/export/excel", series, "chart_data.xlsx");
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd frontend && npx vitest run src/api.test.ts`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add frontend/src/api.ts frontend/src/api.test.ts
git commit -m "feat: add typed API client for login/upload/export"
```

---

## Task 19: Pixel-data calibration math (for live drag recompute)

This is the frontend equivalent of the backend's `calibration.py` — when the user drags a point or an axis marker on the canvas, this converts between screen pixels and chart data values without a server round-trip.

**Files:**
- Create: `frontend/src/calibration.ts`
- Test: `frontend/src/calibration.test.ts`

- [ ] **Step 1: Write the failing test**

`frontend/src/calibration.test.ts`:
```typescript
import { describe, expect, it } from "vitest";
import { fitCalibration, pixelToValue, valueToPixel } from "./calibration";

describe("calibration", () => {
  it("fits a linear mapping from two reference points", () => {
    // y-axis: pixel 300 = value 0, pixel 0 = value 100 (pixels increase downward)
    const calibration = fitCalibration([
      { pixel: 300, value: 0 },
      { pixel: 0, value: 100 },
    ]);

    expect(pixelToValue(calibration, 150)).toBeCloseTo(50, 1);
  });

  it("valueToPixel is the inverse of pixelToValue", () => {
    const calibration = fitCalibration([
      { pixel: 300, value: 0 },
      { pixel: 0, value: 100 },
    ]);

    const pixel = valueToPixel(calibration, 73);
    expect(pixelToValue(calibration, pixel)).toBeCloseTo(73, 1);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npx vitest run src/calibration.test.ts`
Expected: FAIL — `src/calibration.ts` doesn't exist.

- [ ] **Step 3: Implement calibration math**

`frontend/src/calibration.ts`:
```typescript
export interface ReferencePoint {
  pixel: number;
  value: number;
}

export interface Calibration {
  slope: number;
  intercept: number;
}

export function fitCalibration(points: [ReferencePoint, ReferencePoint]): Calibration {
  const [a, b] = points;
  const slope = (b.value - a.value) / (b.pixel - a.pixel);
  const intercept = a.value - slope * a.pixel;
  return { slope, intercept };
}

export function pixelToValue(calibration: Calibration, pixel: number): number {
  return calibration.slope * pixel + calibration.intercept;
}

export function valueToPixel(calibration: Calibration, value: number): number {
  return (value - calibration.intercept) / calibration.slope;
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd frontend && npx vitest run src/calibration.test.ts`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add frontend/src/calibration.ts frontend/src/calibration.test.ts
git commit -m "feat: add frontend pixel-to-data calibration math for live point editing"
```

---

## Task 20: Login page

**Files:**
- Create: `frontend/src/pages/Login.tsx`

- [ ] **Step 1: Implement the login page**

`frontend/src/pages/Login.tsx`:
```tsx
import { FormEvent, useState } from "react";
import { useNavigate } from "react-router-dom";
import { login } from "../api";

export function Login() {
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const navigate = useNavigate();

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      await login(password);
      navigate("/upload");
    } catch {
      setError("Incorrect password");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <form onSubmit={handleSubmit} style={{ maxWidth: 320, margin: "80px auto" }}>
      <h1>Chart Digitizer</h1>
      <input
        type="password"
        value={password}
        onChange={(event) => setPassword(event.target.value)}
        placeholder="Password"
        style={{ width: "100%", padding: 8 }}
      />
      {error && <p style={{ color: "red" }}>{error}</p>}
      <button type="submit" disabled={submitting} style={{ marginTop: 12, width: "100%", padding: 8 }}>
        {submitting ? "Signing in..." : "Sign in"}
      </button>
    </form>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/pages/Login.tsx
git commit -m "feat: add login page"
```

---

## Task 21: Upload page

**Files:**
- Create: `frontend/src/pages/Upload.tsx`

- [ ] **Step 1: Implement the upload page**

`frontend/src/pages/Upload.tsx`:
```tsx
import { ChangeEvent, DragEvent, useState } from "react";
import { useNavigate } from "react-router-dom";
import { uploadChart } from "../api";
import type { UploadResult } from "../types";

interface UploadPageProps {
  onUploaded: (result: UploadResult, imageDataUrl: string) => void;
}

export function Upload({ onUploaded }: UploadPageProps) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();

  async function handleFile(file: File) {
    setError(null);
    setLoading(true);
    try {
      const reader = new FileReader();
      const imageDataUrlPromise = new Promise<string>((resolve) => {
        reader.onload = () => resolve(reader.result as string);
      });
      reader.readAsDataURL(file);

      const result = await uploadChart(file);
      onUploaded(result, await imageDataUrlPromise);
      navigate("/review");
    } catch {
      setError("Couldn't process this image. Try a clearer chart image, or a different file.");
    } finally {
      setLoading(false);
    }
  }

  function handleDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    const file = event.dataTransfer.files[0];
    if (file) handleFile(file);
  }

  function handleFileInput(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (file) handleFile(file);
  }

  return (
    <div style={{ maxWidth: 480, margin: "80px auto", textAlign: "center" }}>
      <h1>Upload a chart image</h1>
      <div
        onDrop={handleDrop}
        onDragOver={(event) => event.preventDefault()}
        style={{ border: "2px dashed #999", borderRadius: 8, padding: 48, cursor: "pointer" }}
      >
        {loading ? (
          <p>Processing...</p>
        ) : (
          <>
            <p>Drag and drop a JPEG or PNG chart image here, or</p>
            <input type="file" accept="image/jpeg,image/png" onChange={handleFileInput} />
          </>
        )}
      </div>
      {error && <p style={{ color: "red" }}>{error}</p>}
    </div>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/pages/Upload.tsx
git commit -m "feat: add upload page with drag-and-drop"
```

---

## Task 22: Interactive chart canvas (draggable points + axis markers)

The core review/correction UI. Renders the uploaded image with Konva, overlays each series' points as draggable circles, and axis calibration reference points as draggable markers along fixed rows/columns near the image edges (exact alignment with the original tick marks isn't required — only the pixel↔value mapping they represent matters for recalibration).

**Files:**
- Create: `frontend/src/components/ChartCanvas.tsx`
- Create: `frontend/src/useHtmlImage.ts`

- [ ] **Step 1: Add a small hook to load the background image for Konva**

`frontend/src/useHtmlImage.ts`:
```typescript
import { useEffect, useState } from "react";

export function useHtmlImage(src: string): HTMLImageElement | null {
  const [image, setImage] = useState<HTMLImageElement | null>(null);

  useEffect(() => {
    const img = new window.Image();
    img.src = src;
    img.onload = () => setImage(img);
  }, [src]);

  return image;
}
```

- [ ] **Step 2: Implement the canvas component**

`frontend/src/components/ChartCanvas.tsx`:
```tsx
import { Circle, Image as KonvaImage, Layer, Rect, Stage } from "react-konva";
import { fitCalibration, pixelToValue, valueToPixel, type Calibration } from "../calibration";
import { useHtmlImage } from "../useHtmlImage";
import type { SeriesData } from "../types";

const SERIES_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"];
const X_MARKER_ROW_OFFSET = 20; // px above the bottom edge of the image
const Y_MARKER_COLUMN_OFFSET = 20; // px right of the left edge of the image

interface ChartCanvasProps {
  imageDataUrl: string;
  series: SeriesData[];
  xReferencePoints: [number, number][];
  yReferencePoints: [number, number][];
  onPointMove: (seriesIndex: number, pointIndex: number, x: number, y: number) => void;
  onDeletePoint: (seriesIndex: number, pointIndex: number) => void;
  onRecalibrate: (axis: "x" | "y", referenceIndex: number, newPixel: number) => void;
}

function bgrToHex([b, g, r]: [number, number, number]): string {
  const toHex = (n: number) => n.toString(16).padStart(2, "0");
  return `#${toHex(r)}${toHex(g)}${toHex(b)}`;
}

export function ChartCanvas({
  imageDataUrl,
  series,
  xReferencePoints,
  yReferencePoints,
  onPointMove,
  onDeletePoint,
  onRecalibrate,
}: ChartCanvasProps) {
  const image = useHtmlImage(imageDataUrl);

  if (!image) {
    return <p>Loading image...</p>;
  }

  const xCalibration: Calibration = fitCalibration([
    { pixel: xReferencePoints[0][0], value: xReferencePoints[0][1] },
    { pixel: xReferencePoints[xReferencePoints.length - 1][0], value: xReferencePoints[xReferencePoints.length - 1][1] },
  ]);
  const yCalibration: Calibration = fitCalibration([
    { pixel: yReferencePoints[0][0], value: yReferencePoints[0][1] },
    { pixel: yReferencePoints[yReferencePoints.length - 1][0], value: yReferencePoints[yReferencePoints.length - 1][1] },
  ]);

  return (
    <Stage width={image.width} height={image.height}>
      <Layer>
        <KonvaImage image={image} />

        {series.map((s, seriesIndex) =>
          s.points.map(([x, y], pointIndex) => (
            <Circle
              key={`${seriesIndex}-${pointIndex}`}
              x={valueToPixel(xCalibration, x)}
              y={valueToPixel(yCalibration, y)}
              radius={5}
              fill={SERIES_COLORS[seriesIndex % SERIES_COLORS.length]}
              stroke="white"
              strokeWidth={1}
              draggable
              onDragEnd={(event) => {
                const newX = pixelToValue(xCalibration, event.target.x());
                const newY = pixelToValue(yCalibration, event.target.y());
                onPointMove(seriesIndex, pointIndex, newX, newY);
              }}
              onDblClick={() => onDeletePoint(seriesIndex, pointIndex)}
            />
          )),
        )}

        {xReferencePoints.map(([pixel], index) => (
          <Rect
            key={`x-ref-${index}`}
            x={pixel - 4}
            y={image.height - X_MARKER_ROW_OFFSET - 4}
            width={8}
            height={8}
            fill="black"
            draggable
            dragBoundFunc={(pos) => ({ x: pos.x, y: image.height - X_MARKER_ROW_OFFSET - 4 })}
            onDragEnd={(event) => onRecalibrate("x", index, event.target.x() + 4)}
          />
        ))}

        {yReferencePoints.map(([pixel], index) => (
          <Rect
            key={`y-ref-${index}`}
            x={Y_MARKER_COLUMN_OFFSET - 4}
            y={pixel - 4}
            width={8}
            height={8}
            fill="black"
            draggable
            dragBoundFunc={(pos) => ({ x: Y_MARKER_COLUMN_OFFSET - 4, y: pos.y })}
            onDragEnd={(event) => onRecalibrate("y", index, event.target.y() + 4)}
          />
        ))}
      </Layer>
    </Stage>
  );
}

export { bgrToHex };
```

- [ ] **Step 3: Verify it compiles**

Run: `cd frontend && npx tsc --noEmit`
Expected: no type errors.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/ChartCanvas.tsx frontend/src/useHtmlImage.ts
git commit -m "feat: add interactive canvas with draggable points and axis calibration markers"
```

---

## Task 23: Data table, add-point control, and Review page

**Files:**
- Create: `frontend/src/components/DataTable.tsx`
- Create: `frontend/src/pages/Review.tsx`

- [ ] **Step 1: Implement the read-only data table**

`frontend/src/components/DataTable.tsx`:
```tsx
import type { SeriesData } from "../types";

interface DataTableProps {
  series: SeriesData[];
}

export function DataTable({ series }: DataTableProps) {
  return (
    <table style={{ borderCollapse: "collapse", width: "100%" }}>
      <thead>
        <tr>
          <th style={{ textAlign: "left", borderBottom: "1px solid #ccc" }}>Series</th>
          <th style={{ textAlign: "left", borderBottom: "1px solid #ccc" }}>X</th>
          <th style={{ textAlign: "left", borderBottom: "1px solid #ccc" }}>Y</th>
        </tr>
      </thead>
      <tbody>
        {series.flatMap((s, seriesIndex) =>
          s.points.map(([x, y], pointIndex) => (
            <tr key={`${seriesIndex}-${pointIndex}`}>
              <td>{s.name}</td>
              <td>{x.toFixed(3)}</td>
              <td>{y.toFixed(3)}</td>
            </tr>
          )),
        )}
      </tbody>
    </table>
  );
}
```

- [ ] **Step 2: Implement the Review page, wiring canvas + table + export**

`frontend/src/pages/Review.tsx`:
```tsx
import { useState } from "react";
import { ChartCanvas } from "../components/ChartCanvas";
import { DataTable } from "../components/DataTable";
import { exportCsv, exportExcel } from "../api";
import type { SeriesData, UploadResult } from "../types";

interface ReviewPageProps {
  uploadResult: UploadResult;
  imageDataUrl: string;
}

export function Review({ uploadResult, imageDataUrl }: ReviewPageProps) {
  const [series, setSeries] = useState<SeriesData[]>(uploadResult.series);
  const [xReferencePoints, setXReferencePoints] = useState(uploadResult.xReferencePoints);
  const [yReferencePoints, setYReferencePoints] = useState(uploadResult.yReferencePoints);

  function handlePointMove(seriesIndex: number, pointIndex: number, x: number, y: number) {
    setSeries((prev) =>
      prev.map((s, i) =>
        i !== seriesIndex
          ? s
          : { ...s, points: s.points.map((p, j) => (j === pointIndex ? [x, y] : p)) as [number, number][] },
      ),
    );
  }

  function handleDeletePoint(seriesIndex: number, pointIndex: number) {
    setSeries((prev) =>
      prev.map((s, i) =>
        i !== seriesIndex ? s : { ...s, points: s.points.filter((_, j) => j !== pointIndex) },
      ),
    );
  }

  function handleAddPoint(seriesIndex: number) {
    const seriesPoints = series[seriesIndex].points;
    const lastPoint = seriesPoints[seriesPoints.length - 1] ?? [0, 0];
    setSeries((prev) =>
      prev.map((s, i) => (i !== seriesIndex ? s : { ...s, points: [...s.points, [lastPoint[0], lastPoint[1]]] })),
    );
  }

  function handleRecalibrate(axis: "x" | "y", referenceIndex: number, newPixel: number) {
    const setter = axis === "x" ? setXReferencePoints : setYReferencePoints;
    setter((prev) => prev.map((ref, i) => (i === referenceIndex ? [newPixel, ref[1]] : ref)));
  }

  return (
    <div style={{ display: "flex", gap: 24, padding: 24 }}>
      <div>
        <ChartCanvas
          imageDataUrl={imageDataUrl}
          series={series}
          xReferencePoints={xReferencePoints}
          yReferencePoints={yReferencePoints}
          onPointMove={handlePointMove}
          onDeletePoint={handleDeletePoint}
          onRecalibrate={handleRecalibrate}
        />
        <p style={{ color: "#666", fontSize: 13 }}>
          Drag a point to correct it. Double-click a point to delete it. Drag a black square to
          recalibrate that axis reference.
        </p>
        {series.map((s, i) => (
          <button key={s.name} onClick={() => handleAddPoint(i)}>
            + Add point to {s.name}
          </button>
        ))}
      </div>
      <div style={{ flex: 1 }}>
        <DataTable series={series} />
        <div style={{ marginTop: 16, display: "flex", gap: 8 }}>
          <button onClick={() => exportCsv(series)}>Download CSV</button>
          <button onClick={() => exportExcel(series)}>Download Excel</button>
          <a
            href={`data:image/png;base64,${uploadResult.overlayImageBase64}`}
            download="chart_overlay.png"
          >
            <button>Download overlay PNG</button>
          </a>
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 3: Verify it compiles**

Run: `cd frontend && npx tsc --noEmit`
Expected: no type errors.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/DataTable.tsx frontend/src/pages/Review.tsx
git commit -m "feat: add data table and review page with export"
```

---

## Task 24: App routing and manual browser verification

**Files:**
- Modify: `frontend/src/App.tsx`, `frontend/src/main.tsx`

- [ ] **Step 1: Wire up routing and shared upload-result state**

`frontend/src/App.tsx`:
```tsx
import { useState } from "react";
import { Navigate, Route, Routes } from "react-router-dom";
import { Login } from "./pages/Login";
import { Upload } from "./pages/Upload";
import { Review } from "./pages/Review";
import type { UploadResult } from "./types";

export default function App() {
  const [uploadResult, setUploadResult] = useState<UploadResult | null>(null);
  const [imageDataUrl, setImageDataUrl] = useState<string>("");

  function handleUploaded(result: UploadResult, dataUrl: string) {
    setUploadResult(result);
    setImageDataUrl(dataUrl);
  }

  return (
    <Routes>
      <Route path="/" element={<Login />} />
      <Route path="/upload" element={<Upload onUploaded={handleUploaded} />} />
      <Route
        path="/review"
        element={uploadResult ? <Review uploadResult={uploadResult} imageDataUrl={imageDataUrl} /> : <Navigate to="/upload" />}
      />
    </Routes>
  );
}
```

`frontend/src/main.tsx`:
```tsx
import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import App from "./App";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </React.StrictMode>,
);
```

- [ ] **Step 2: Run the full frontend test suite**

Run: `cd frontend && npx vitest run`
Expected: all tests PASS.

- [ ] **Step 3: Manual browser verification**

Run: `cd backend && APP_PASSWORD=devpass SESSION_SECRET_KEY=devsecret uvicorn app.main:app --reload --port 8000` (separate terminal)
Run: `cd frontend && VITE_API_BASE_URL=http://localhost:8000 npm run dev`

In a browser:
1. Log in with `devpass`.
2. Upload a real Kaplan-Meier curve image (e.g. a screenshot from a published trial figure) and a simple line/bar chart.
3. Confirm the review screen shows the image with overlaid points roughly on the curve/bars.
4. Drag a point and confirm the data table updates live with a plausible new value.
5. Drag a black calibration square and confirm all points on that axis shift accordingly.
6. Click "Download CSV" and confirm the file opens with sane values.

Note any systematic misdetection (e.g. always off by one tick, wrong chart type) — that indicates which pipeline module (Tasks 3-13) needs threshold tuning against real (non-synthetic) images, since all backend tests so far only used matplotlib-generated fixtures.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/App.tsx frontend/src/main.tsx
git commit -m "feat: wire up app routing between login, upload, and review pages"
```

---
