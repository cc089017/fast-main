import { useEffect, useRef, useState } from "react";
import { predictFaceFrame, uploadFaceFrame } from "../api/measure";
import {
  initFaceLandmarker,
  detectOnVideo,
  drawSpecificPoints,
  FEATURE_POINT_INDEXES,
  computeBasicFeatures,
} from "../lib/faceLandmarker";

export default function FaceMeasure() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const landmarkerRef = useRef(null);

  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const [preview, setPreview] = useState(null); // { pred_proba, pred_label, features }
  const [saved, setSaved] = useState(null);     // { face_id, result_text, created_at, ... }

  useEffect(() => {
    (async () => {
      try {
        const s = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "user" }, audio: false
        });
        if (videoRef.current) {
          videoRef.current.srcObject = s;
          await videoRef.current.play();
        }
        landmarkerRef.current = await initFaceLandmarker();
      } catch {
        setErr("카메라/랜드마커 초기화 실패");
      }
    })();
  }, []);

  const drawGuides = (ctx, w, h) => {
    ctx.save(); ctx.strokeStyle="rgba(255,255,255,0.9)"; ctx.lineWidth=2;
    ctx.beginPath(); ctx.moveTo(w/2, 0); ctx.lineTo(w/2, h); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, h*0.35); ctx.lineTo(w, h*0.35); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, h*0.65); ctx.lineTo(w, h*0.65); ctx.stroke();
    ctx.restore();
  };

  const onCapture = async () => {
    setErr(""); setPreview(null); setSaved(null);

    const v = videoRef.current, c = canvasRef.current, landmarker = landmarkerRef.current;
    if (!v || !c || !landmarker) return;

    const w = v.videoWidth, h = v.videoHeight;
    c.width = w; c.height = h;
    const ctx = c.getContext("2d");

    // 배경 프레임 (표시용 미러)
    ctx.save(); ctx.translate(w,0); ctx.scale(-1,1); ctx.drawImage(v,0,0,w,h); ctx.restore();
    drawGuides(ctx, w, h);

    // 랜드마크
    const det = detectOnVideo(landmarker, v, performance.now());
    if (!det?.faceLandmarks?.length) { setErr("얼굴을 인식하지 못했습니다."); return; }
    const lm = det.faceLandmarks[0];

    // 점찍기(미러로 표시)
    drawSpecificPoints(ctx, lm, w, h, FEATURE_POINT_INDEXES, { mirror: true, radius: 1.5 });

    // 특징치(원본 좌표계 기반) — 저장용 result_text 생성에 사용됨
    const features = computeBasicFeatures(lm);

    setBusy(true);
    try {
      const blob = await new Promise((res) => c.toBlob(res, "image/jpeg", 0.9));
      if (!blob) throw new Error("캔버스 이미지 생성 실패");

      // ① 예측 전용 호출 → 화면 즉시 표시
      const pred = await predictFaceFrame(blob); // { pred_proba, pred_label, features }
      setPreview(pred);

      // ② 저장 호출 → DB 저장 및 result_text 수신/표시
      const savedRec = await uploadFaceFrame(blob, {
        predLabel: Number(pred?.pred_label),
        features, // 서버가 부위명 이유문장 만들 수 있게
      });
      setSaved(savedRec);
    } catch (e) {
      setErr(e?.response?.data?.detail || e.message || "업로드/예측 실패");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="w-full min-h-screen flex flex-col items-center px-6 py-10">
      <h1 className="text-3xl font-extrabold mb-6">F.A.S.T — Face 측정</h1>
      <div className="grid md:grid-cols-2 gap-8 w-full max-w-6xl">
        {/* 좌측: 카메라/캡처 */}
        <div className="space-y-4">
          <video ref={videoRef} className="w-full rounded-2xl shadow" muted playsInline />
          <canvas ref={canvasRef} className="w-full rounded-2xl shadow" />
          <button
            onClick={onCapture}
            disabled={busy}
            className="px-6 py-3 rounded-2xl bg-blue-600 text-white font-semibold hover:bg-blue-700 disabled:opacity-50"
          >
            {busy ? "처리 중..." : "측정/예측/저장"}
          </button>
          {err && <p className="text-red-600">{err}</p>}
        </div>

        {/* 우측: 상단 = 모델 미리보기 / 하단 = 저장 결과 */}
        <div className="space-y-6">
          <section className="p-5 rounded-2xl border shadow bg-white">
            <h2 className="text-lg font-bold mb-2">모델 미리보기 (저장 안 함)</h2>
            {!preview ? (
              <p className="text-gray-500">측정 후 바로 예측 결과가 표시됩니다.</p>
            ) : (
              <>
                <div className="text-sm space-y-1">
                  <div><b>확률(1):</b> {Number(preview.pred_proba).toFixed(3)}</div>
                  <div><b>판정:</b> {Number(preview.pred_label) === 1 ? "비정상(의심)" : "정상"}</div>
                </div>
                <details className="mt-3">
                  <summary className="cursor-pointer text-blue-600">미리보기 features</summary>
                  <pre className="text-xs bg-gray-50 p-3 rounded-lg overflow-x-auto">
                    {JSON.stringify(preview.features, null, 2)}
                  </pre>
                </details>
              </>
            )}
          </section>

          <section className="p-5 rounded-2xl border shadow bg-white">
            <h2 className="text-lg font-bold mb-2">저장 결과 (DB)</h2>
            {!saved ? (
              <p className="text-gray-500">저장 후 결과문이 표시됩니다.</p>
            ) : (
              <>
                <p className="text-base leading-relaxed whitespace-pre-line">
                  {saved.result_text}
                </p>
                <p className="text-xs text-gray-500 mt-2">
                  face_id: {saved.face_id}{saved.created_at ? ` · ${new Date(saved.created_at).toLocaleString()}` : ""}
                </p>
              </>
            )}
          </section>
        </div>
      </div>
    </div>
  );
}
