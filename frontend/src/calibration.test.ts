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

  it("fits a genuine least-squares line over 3+ non-collinear reference points", () => {
    // Roughly value = 2 * pixel, with jitter on an off-center point (using
    // unevenly spaced pixels so the jitter actually perturbs the OLS slope,
    // not just the intercept - a symmetric midpoint jitter with evenly
    // spaced pixels leaves the OLS slope unchanged, which would make this
    // test degenerate).
    const points = [
      { pixel: 0, value: 0 },
      { pixel: 5, value: 16 }, // jittered off the y=2x line
      { pixel: 20, value: 40 },
    ];

    const calibration = fitCalibration(points);

    // Independently compute the closed-form OLS solution and confirm the
    // implementation matches it exactly (not, say, a 2-point fit through
    // the first and last points, which would give slope=2, intercept=0).
    const n = points.length;
    const meanPixel = points.reduce((s, p) => s + p.pixel, 0) / n;
    const meanValue = points.reduce((s, p) => s + p.value, 0) / n;
    let covariance = 0;
    let variance = 0;
    for (const p of points) {
      covariance += (p.pixel - meanPixel) * (p.value - meanValue);
      variance += (p.pixel - meanPixel) ** 2;
    }
    const expectedSlope = covariance / variance;
    const expectedIntercept = meanValue - expectedSlope * meanPixel;

    expect(calibration.slope).toBeCloseTo(expectedSlope, 10);
    expect(calibration.intercept).toBeCloseTo(expectedIntercept, 10);

    // Sanity check: this must NOT equal the naive first/last 2-point fit
    // (slope 2, intercept 0), since the middle point is jittered off that line.
    expect(calibration.slope).not.toBeCloseTo(2, 2);

    // The least-squares fit must minimize total squared error at least as
    // well as the naive first/last-point fit.
    const sumSquaredError = (cal: { slope: number; intercept: number }) =>
      points.reduce((sum, p) => sum + (p.value - (cal.slope * p.pixel + cal.intercept)) ** 2, 0);
    const naiveFit = { slope: 2, intercept: 0 };
    expect(sumSquaredError(calibration)).toBeLessThan(sumSquaredError(naiveFit));
  });

  it("a 2-point OLS fit reduces to the same simple line as the original 2-point implementation", () => {
    const calibration = fitCalibration([
      { pixel: 50, value: 10 },
      { pixel: 150, value: 210 },
    ]);

    // Slope should equal (210-10)/(150-50) = 2, intercept = 10 - 2*50 = -90
    expect(calibration.slope).toBeCloseTo(2, 10);
    expect(calibration.intercept).toBeCloseTo(-90, 10);
  });

  it("valueToPixel remains the inverse of pixelToValue with an N-point fit", () => {
    const calibration = fitCalibration([
      { pixel: 0, value: 5 },
      { pixel: 10, value: 18 },
      { pixel: 25, value: 51 },
      { pixel: 40, value: 78 },
    ]);

    for (const value of [5, 18, 51, 78, 33]) {
      const pixel = valueToPixel(calibration, value);
      expect(pixelToValue(calibration, pixel)).toBeCloseTo(value, 6);
    }
  });
});
