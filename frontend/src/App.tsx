import { useState } from "react";
import { Navigate, Route, Routes } from "react-router-dom";
import { Login } from "./pages/Login";
import { Upload } from "./pages/Upload";
import { Review } from "./pages/Review";
import type { UploadResult } from "./types";

export default function App() {
  const [uploadResult, setUploadResult] = useState<UploadResult | null>(null);
  const [imageDataUrl, setImageDataUrl] = useState<string>("");

  function handleUploaded(result: UploadResult, dataUrl: string) {
    setUploadResult(result);
    setImageDataUrl(dataUrl);
  }

  return (
    <Routes>
      <Route path="/" element={<Login />} />
      <Route path="/upload" element={<Upload onUploaded={handleUploaded} />} />
      <Route
        path="/review"
        element={uploadResult ? <Review uploadResult={uploadResult} imageDataUrl={imageDataUrl} /> : <Navigate to="/upload" />}
      />
    </Routes>
  );
}
