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
