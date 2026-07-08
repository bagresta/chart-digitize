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
