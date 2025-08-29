// front-end/src/api/measure.js
import http from "@/lib/http";

export async function predictFaceFrame(blob) {
  const fd = new FormData();
  fd.append("file", blob, "frame.jpg"); // 예측 전용 엔드포인트는 'file'
  const res = await http.post("/api/v1/measure/face/predict", fd, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return res.data; // { modality, pred_proba, pred_label, features }
}

export async function uploadFaceFrame(blob, { predLabel, features } = {}) {
  const fd = new FormData();
  fd.append("image", blob, "frame.jpg"); // 저장 엔드포인트는 'image'
  if (typeof predLabel === "number") fd.append("pred_label", String(predLabel));
  if (features) fd.append("features_json", JSON.stringify(features));
  const res = await http.post("/api/v1/measure/face/upload", fd, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return res.data; // FaceOut
}
