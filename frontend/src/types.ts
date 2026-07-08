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
