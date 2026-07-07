# Chart Digitizer — Design Spec

Date: 2026-07-07

## Purpose

A hosted web app that takes an uploaded chart image (JPEG/PNG) — general charts
(line, scatter, bar) and, in particular, Kaplan-Meier survival curves from
clinical trial figures — and automatically extracts axis calibration and
per-series data points, so the numbers can be reused (e.g. in economic models
for HTA/PBAC submissions). Extraction is fully automatic; a review screen lets
the user visually correct any mistakes before export.

## Non-goals (deferred to a future version)

- Batch/multi-image processing
- Saved projects / history / accounts beyond a single shared password
- Automatic reconstruction of individual patient data (e.g. Guyot algorithm)
  from KM curves — only digitized (x, y) points are produced, not IPD

## Architecture

Two services, both free-tier hosted:

- **Frontend**: React SPA (Vite), deployed to Vercel. Canvas-based image
  viewer/editor (Konva or Fabric.js) for the review/correction step.
- **Backend**: Python FastAPI, deployed to Render. Runs the extraction
  pipeline, including OCR via Tesseract for label reading. Stateless —
  uploaded images are processed in memory per-request and never persisted to
  disk or a database.

Rationale: OpenCV's Python ecosystem is the most capable option for the
geometric image-processing work (axis/line detection, color-based curve
tracing), so the backend is Python. Precise drag-and-drop point correction is
easiest to build well with a JS canvas library, so the frontend is a React
SPA rather than a server-rendered page. Splitting the two costs an extra
deploy but avoids compromising either half.

## Extraction pipeline (backend, per uploaded image)

1. **OpenCV geometric pass** — detect the plot bounding box and axis lines
   (Hough transform), locate tick mark pixel positions along each axis.
2. **OCR pass (Tesseract)** — crop and preprocess the regions around each
   axis (upscale, binarize/threshold, deskew) to maximize read accuracy, then
   OCR the tick label text and axis titles. Chart type (line / scatter / bar
   / Kaplan-Meier) and log-scale detection are inferred heuristically from
   the plotted shapes (e.g. step patterns for KM, discrete bars for bar
   charts) rather than read from text. Legend entries (series name + color
   swatch) are located via layout heuristics (small color swatch immediately
   left of a text run) and OCR'd the same way.
3. **Calibration** — match OCR'd tick values to detected tick pixel positions
   in order; fit a pixel→data transform (linear or log) per axis.
4. **Curve/point extraction** — color-cluster the plot area using legend
   colors as hints to isolate each series' pixels.
   - Line / KM charts: trace one y (or step) value per x pixel-column; KM
     curves are snapped to a step function, and censoring tick marks are
     detected as a separate marker set per series.
   - Scatter charts: connected-component blob centroids.
   - Bar charts: bar extents (top edge per bar).
5. **Output** — per-series (x, y) data table in real data units, plus an
   overlay PNG (original image with detected points/curve drawn on top) for
   visual QC.

## UI/UX flow

1. **Login gate** — single shared password (backend secret env var), issues a
   short-lived signed session cookie; all API routes require it. No user
   accounts.
2. **Upload screen** — drag-and-drop or file picker (JPEG/PNG). Spinner while
   the pipeline runs.
3. **Review screen** (core interaction):
   - Canvas shows the original image with detected axis calibration markers
     and extracted points/curve overlaid per series, color-coded to match
     the legend.
   - Drag any data point to correct its position; the corresponding data
     value recomputes live from the pixel↔data calibration.
   - Drag axis calibration markers to recalibrate if auto-detected tick
     alignment looks wrong — recalculates all points in that series live.
   - Add/delete point controls for missed or spurious auto-detected points.
   - A read-only data table below the canvas mirrors current points per
     series in real time.
4. **Export** — download CSV (per-series or combined with a series column)
   and Excel; the overlay PNG is also downloadable as a QC record.

## Error handling

- Unsupported/corrupt file → inline error, no crash.
- OCR fails to read tick labels confidently (low confidence score, or text
  doesn't parse as numbers) → surfaced as "couldn't read axis labels
  automatically"; a minimal manual override lets the user type in axis
  min/max values as a fallback so they are not fully blocked. Given OCR is
  less reliable than a vision LLM on stylized/low-res chart text, expect this
  fallback to be used more often — worth validating during testing how often
  it triggers on real figures.
- No axis lines detected (e.g. borderless chart) → explicit error rather than
  silently producing garbage data.

## Auth & secrets

- Shared app password stored as an environment variable on Render; never
  exposed to the frontend bundle. No external API key is required — OCR
  (Tesseract) runs entirely on the backend server, no LLM API dependency.
- Simple password form → backend validates → issues signed session token
  (cookie) → required on all API routes.

## Testing / validation

Before considering this done, verify against real chart types:

- A synthetic line chart with known data
- A scatter chart with known data
- A bar chart with known data
- A real Kaplan-Meier curve image with multiple arms — confirm exported
  values are numerically close to known-correct reference values

## Hosting cost

Vercel and Render free tiers cover this usage pattern (low-volume, on-demand
use). With OCR running locally on the backend instead of calling an external
LLM API, there is no per-image API charge and no external API key to manage
— hosting is effectively free.
