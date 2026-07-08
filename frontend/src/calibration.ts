export interface ReferencePoint {
  pixel: number;
  value: number;
}

export interface Calibration {
  slope: number;
  intercept: number;
}

/**
 * Fits value = slope * pixel + intercept via ordinary least squares over an
 * arbitrary number of reference points (2 or more). Mirrors the simple OLS
 * fit in backend/app/pipeline/calibration.py's fit_axis_calibration, minus
 * the backend's outlier-rejection step (that's an OCR-time safety net; here
 * we just fit whatever reference points we're given, including
 * user-corrected ones).
 */
export function fitCalibration(points: ReferencePoint[]): Calibration {
  if (points.length < 2) {
    throw new Error("fitCalibration requires at least 2 reference points");
  }

  const n = points.length;
  const meanPixel = points.reduce((sum, p) => sum + p.pixel, 0) / n;
  const meanValue = points.reduce((sum, p) => sum + p.value, 0) / n;

  let covariance = 0;
  let variance = 0;
  for (const p of points) {
    const dPixel = p.pixel - meanPixel;
    covariance += dPixel * (p.value - meanValue);
    variance += dPixel * dPixel;
  }

  const slope = covariance / variance;
  const intercept = meanValue - slope * meanPixel;
  return { slope, intercept };
}

export function pixelToValue(calibration: Calibration, pixel: number): number {
  return calibration.slope * pixel + calibration.intercept;
}

export function valueToPixel(calibration: Calibration, value: number): number {
  return (value - calibration.intercept) / calibration.slope;
}
