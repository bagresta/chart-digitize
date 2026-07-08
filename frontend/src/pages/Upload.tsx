import { type ChangeEvent, type DragEvent, useState } from "react";
import { useNavigate } from "react-router-dom";
import { uploadChart } from "../api";
import type { UploadResult } from "../types";

interface UploadPageProps {
  onUploaded: (result: UploadResult, imageDataUrl: string) => void;
}

export function Upload({ onUploaded }: UploadPageProps) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();

  async function handleFile(file: File) {
    setError(null);
    setLoading(true);
    try {
      const reader = new FileReader();
      const imageDataUrlPromise = new Promise<string>((resolve) => {
        reader.onload = () => resolve(reader.result as string);
      });
      reader.readAsDataURL(file);

      const result = await uploadChart(file);
      onUploaded(result, await imageDataUrlPromise);
      navigate("/review");
    } catch {
      setError("Couldn't process this image. Try a clearer chart image, or a different file.");
    } finally {
      setLoading(false);
    }
  }

  function handleDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    const file = event.dataTransfer.files[0];
    if (file) handleFile(file);
  }

  function handleFileInput(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (file) handleFile(file);
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
      {error && <p style={{ color: "red" }}>{error}</p>}
    </div>
  );
}
