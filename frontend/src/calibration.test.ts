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
