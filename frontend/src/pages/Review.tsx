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
          : {
              ...s,
              points: s.points.map((p, j) => (j === pointIndex ? ([x, y] as [number, number]) : p)),
            },
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
      prev.map((s, i) =>
        i !== seriesIndex
          ? s
          : { ...s, points: [...s.points, [lastPoint[0], lastPoint[1]] as [number, number]] },
      ),
    );
  }

  function handleRecalibrate(axis: "x" | "y", referenceIndex: number, newPixel: number) {
    const setter = axis === "x" ? setXReferencePoints : setYReferencePoints;
    setter((prev) =>
      prev.map((ref, i) => (i === referenceIndex ? ([newPixel, ref[1]] as [number, number]) : ref)),
    );
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
