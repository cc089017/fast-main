import { apiFetch } from "./http";
const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

export async function uploadFaceFrame(blob) {
  const fd = new FormData();
  fd.append("file", blob, "frame.jpg");
  return apiFetch(`${API_BASE}/api/v1/measure/face/upload`, { method: "POST", body: fd });
}
