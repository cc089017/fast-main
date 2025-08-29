// 사진 + 결과(정상/비정상) + features(JSON)를 DB 저장 (face 라우터)
import http from "@/lib/http";

export async function saveFaceRecord({ file, resultText, features }) {
  const fd = new FormData();
  fd.append("image", file, "face.jpg");
  fd.append("result_text", resultText); // "정상" | "비정상"
  if (features) fd.append("landmarks_json", JSON.stringify(features));

  const res = await http.post("/api/v1/face", fd, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return res.data; // { face_id, user_id, result_text, created_at }
}
