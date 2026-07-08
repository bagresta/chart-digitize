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
