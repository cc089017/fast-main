import { useEffect, useRef, useState } from "react";
import { uploadFaceFrame } from "../api/measure";

export default function FaceMeasure() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState(null);
  const [err, setErr] = useState("");

  useEffect(() => {
    (async () => {
      try {
        const s = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" }, audio: false });
        if (videoRef.current) { videoRef.current.srcObject = s; await videoRef.current.play(); }
      } catch { setErr("카메라 접근 실패"); }
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
    setErr("");
    const v = videoRef.current, c = canvasRef.current;
    if (!v || !c) return;
    const w = v.videoWidth, h = v.videoHeight;
    c.width = w; c.height = h;
    const ctx = c.getContext("2d");
    ctx.save(); ctx.translate(w,0); ctx.scale(-1,1); ctx.drawImage(v,0,0,w,h); ctx.restore();
    drawGuides(ctx,w,h);

    setBusy(true);
    try {
      const blob = await new Promise(res => c.toBlob(res, "image/jpeg", 0.9));
      const data = await uploadFaceFrame(blob);
      setResult(data);
    } catch (e) { setErr(e.message || "업로드 실패"); }
    finally { setBusy(false); }
  };

  return (
    <div className="w-full min-h-screen flex flex-col items-center px-6 py-10">
      <h1 className="text-3xl font-extrabold mb-6">F.A.S.T — Face 측정</h1>
      <div className="grid md:grid-cols-2 gap-8 w-full max-w-6xl">
        <div className="space-y-4">
          <video ref={videoRef} className="w-full rounded-2xl shadow" muted playsInline />
          <canvas ref={canvasRef} className="w-full rounded-2xl shadow" />
          <button
            onClick={onCapture} disabled={busy}
            className="px-6 py-3 rounded-2xl bg-blue-600 text-white font-semibold hover:bg-blue-700 disabled:opacity-50"
          >{busy ? "측정 중..." : "측정 시작"}</button>
          {err && <p className="text-red-600">{err}</p>}
        </div>
        <div className="space-y-4">
          <h2 className="text-xl font-bold">결과</h2>
          {!result ? <p className="text-gray-500">측정 후 결과가 표시됩니다.</p> : (
            <div className="p-4 rounded-2xl border shadow">
              <div className="text-sm space-y-1">
                <div><b>확률(1):</b> {result.pred_proba.toFixed(3)}</div>
                <div><b>판정:</b> {result.pred_label === 1 ? "비정상(의심)" : "정상"}</div>
              </div>
              <details className="mt-3">
                <summary className="cursor-pointer text-blue-600">features 보기</summary>
                <pre className="text-xs bg-gray-50 p-3 rounded-lg overflow-x-auto">
                  {JSON.stringify(result.features, null, 2)}
                </pre>
              </details>
              <p className="text-xs text-gray-500 mt-3">※ 참고용 결과이며 의료적 진단이 아닙니다.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
