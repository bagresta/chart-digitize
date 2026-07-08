import { type ChangeEvent, type DragEvent, type FormEvent, useState } from "react";
import { useNavigate } from "react-router-dom";
import { AxisCalibrationFailedError, uploadChart } from "../api";
import type { UploadResult } from "../types";

interface UploadPageProps {
  onUploaded: (result: UploadResult, imageDataUrl: string) => void;
}

export function Upload({ onUploaded }: UploadPageProps) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [needsManualAxis, setNeedsManualAxis] = useState(false);
  const [pendingFile, setPendingFile] = useState<File | null>(null);
  const [manualRange, setManualRange] = useState({ xMin: "0", xMax: "1", yMin: "0", yMax: "1" });
  const navigate = useNavigate();

  async function processFile(file: File, useManualRange: boolean) {
    setError(null);
    setLoading(true);
    try {
      const reader = new FileReader();
      const imageDataUrlPromise = new Promise<string>((resolve) => {
        reader.onload = () => resolve(reader.result as string);
      });
      reader.readAsDataURL(file);

      const result = await uploadChart(
        file,
        useManualRange
          ? {
              xMin: Number(manualRange.xMin),
              xMax: Number(manualRange.xMax),
              yMin: Number(manualRange.yMin),
              yMax: Number(manualRange.yMax),
            }
          : undefined,
      );
      onUploaded(result, await imageDataUrlPromise);
      navigate("/review");
    } catch (err) {
      if (err instanceof AxisCalibrationFailedError) {
        setPendingFile(file);
        setNeedsManualAxis(true);
        setError(err.message);
      } else {
        setError("Couldn't process this image. Try a clearer chart image, or a different file.");
      }
    } finally {
      setLoading(false);
    }
  }

  function handleDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    const file = event.dataTransfer.files[0];
    if (file) processFile(file, false);
  }

  function handleFileInput(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (file) processFile(file, false);
  }

  function handleManualSubmit(event: FormEvent) {
    event.preventDefault();
    if (pendingFile) processFile(pendingFile, true);
  }

  if (needsManualAxis) {
    return (
      <form onSubmit={handleManualSubmit} style={{ maxWidth: 320, margin: "80px auto" }}>
        <h2>Couldn't read axis labels automatically</h2>
        <p>{error}</p>
        <label>X min <input value={manualRange.xMin} onChange={(e) => setManualRange({ ...manualRange, xMin: e.target.value })} /></label>
        <label>X max <input value={manualRange.xMax} onChange={(e) => setManualRange({ ...manualRange, xMax: e.target.value })} /></label>
        <label>Y min <input value={manualRange.yMin} onChange={(e) => setManualRange({ ...manualRange, yMin: e.target.value })} /></label>
        <label>Y max <input value={manualRange.yMax} onChange={(e) => setManualRange({ ...manualRange, yMax: e.target.value })} /></label>
        <button type="submit" disabled={loading}>{loading ? "Processing..." : "Continue"}</button>
      </form>
    );
  }

  return (
    <div style={{ maxWidth: 480, margin: "80px auto", textAlign: "center" }}>
      <h1>Upload a chart image</h1>
      <div
        onDrop={handleDrop}
        onDragOver={(event) => event.preventDefault()}
        style={{ border: "2px dashed #999", borderRadius: 8, padding: 48, cursor: "pointer" }}
      >
        {loading ? (
          <p>Processing...</p>
        ) : (
          <>
            <p>Drag and drop a JPEG or PNG chart image here, or</p>
            <input type="file" accept="image/jpeg,image/png" onChange={handleFileInput} />
          </>
        )}
      </div>
      {error && !needsManualAxis && <p style={{ color: "red" }}>{error}</p>}
    </div>
  );
}
