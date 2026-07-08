import type { SeriesData } from "../types";

interface DataTableProps {
  series: SeriesData[];
}

export function DataTable({ series }: DataTableProps) {
  return (
    <table style={{ borderCollapse: "collapse", width: "100%" }}>
      <thead>
        <tr>
          <th style={{ textAlign: "left", borderBottom: "1px solid #ccc" }}>Series</th>
          <th style={{ textAlign: "left", borderBottom: "1px solid #ccc" }}>X</th>
          <th style={{ textAlign: "left", borderBottom: "1px solid #ccc" }}>Y</th>
        </tr>
      </thead>
      <tbody>
        {series.flatMap((s, seriesIndex) =>
          s.points.map(([x, y], pointIndex) => (
            <tr key={`${seriesIndex}-${pointIndex}`}>
              <td>{s.name}</td>
              <td>{x.toFixed(3)}</td>
              <td>{y.toFixed(3)}</td>
            </tr>
          )),
        )}
      </tbody>
    </table>
  );
}
