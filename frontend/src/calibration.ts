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
