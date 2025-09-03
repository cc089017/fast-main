import { useEffect, useRef, useState } from "react";
import { predictFaceFrame, uploadFaceFrame } from "../api/measure";
import {
  initFaceLandmarker,
  detectOnVideo,
  computeBasicFeatures,
  drawSpecificPoints,
  FEATURE_POINT_INDEXES,
} from "../lib/faceLandmarker";

/**
 * 단순 정면 판정:
 * - 눈 높이 차이(roll) ≤ 5% 화면 높이
 * - 코 위치(x)와 눈 중앙 차이 ≤ 5% 화면 너비
 * - 턱 높이는 화면 65% 근처
 */

export default function FaceMeasure() {
  const videoRef = useRef(null);
  const overlayRef = useRef(null);
  const landmarkerRef = useRef(null);
  const animRef = useRef(null);

  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const [saved, setSaved] = useState(null);
  const [alignOK, setAlignOK] = useState(false);

  useEffect(() => {
    let running = true;
    (async () => {
      try {
        const s = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "user" },
          audio: false,
        });
        if (videoRef.current) {
          videoRef.current.srcObject = s;
          await videoRef.current.play();
        }
        landmarkerRef.current = await initFaceLandmarker();
      } catch {
        setErr("카메라 초기화 실패");
        return;
      }

      const loop = () => {
        if (!running) return;
        const v = videoRef.current;
        const c = overlayRef.current;
        const landmarker = landmarkerRef.current;

        if (!v || !c || !landmarker || !v.videoWidth || !v.videoHeight) {
          animRef.current = requestAnimationFrame(loop);
          return;
        }

        const w = v.videoWidth;
        const h = v.videoHeight;
        if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }

        const det = detectOnVideo(landmarker, v, performance.now());
        const lm = det?.faceLandmarks?.[0];
        const ctx = c.getContext("2d");
        ctx.clearRect(0, 0, w, h);

        let ok = false;

        if (lm) {
          // 기준 포인트
          const leftEye = lm[33], rightEye = lm[263], nose = lm[1], chin = lm[152];
          if (leftEye && rightEye && nose && chin) {
            const eyeDiff = Math.abs(leftEye.y - rightEye.y) * h;
            const rollOk = eyeDiff < h * 0.05;

            const midEyeX = (leftEye.x + rightEye.x) / 2 * w;
            const noseX = nose.x * w;
            const yawOk = Math.abs(noseX - midEyeX) < w * 0.05;

            const chinY = chin.y * h;
            const chinTarget = h * 0.63;
            const chinOk = Math.abs(chinY - chinTarget) < h * 0.05;

            ok = rollOk && yawOk && chinOk;

            // 오버레이 그리기
            ctx.strokeStyle = ok ? "lime" : "red";
            ctx.setLineDash([6, 6]);
            ctx.lineWidth = 2;

            // 중앙 수직선
            ctx.beginPath();
            ctx.moveTo(w / 2, 0);
            ctx.lineTo(w / 2, h);
            ctx.stroke();

            // 턱선 가이드
            ctx.beginPath();
            ctx.moveTo(0, chinTarget);
            ctx.lineTo(w, chinTarget);
            ctx.stroke();

            // 상태 텍스트
            ctx.setLineDash([]);
            ctx.font = "18px sans-serif";
            ctx.fillStyle = ok ? "lime" : "red";
            ctx.fillText(ok ? "정면 OK" : "맞춰주세요", 20, 30);
          }
        }
        setAlignOK(ok);

        animRef.current = requestAnimationFrame(loop);
      };
      animRef.current = requestAnimationFrame(loop);
    })();

    return () => {
      running = false;
      if (animRef.current) cancelAnimationFrame(animRef.current);
      const v = videoRef.current;
      const stream = v?.srcObject;
      if (stream) stream.getTracks().forEach(t => t.stop());
    };
  }, []);

  // 캡처 & 저장 (비디오 + 점만)
  const onCapture = async () => {
    setErr("");
    setSaved(null);

    const v = videoRef.current;
    const landmarker = landmarkerRef.current;
    if (!v || !landmarker) return;

    const w = v.videoWidth, h = v.videoHeight;
    const det = detectOnVideo(landmarker, v, performance.now());
    if (!det?.faceLandmarks?.length) { setErr("얼굴 인식 실패"); return; }
    const lm = det.faceLandmarks[0];
    const features = computeBasicFeatures(lm);

    const off = document.createElement("canvas");
    off.width = w; off.height = h;
    const octx = off.getContext("2d");

    octx.save();
    octx.translate(w, 0);
    octx.scale(-1, 1);
    octx.drawImage(v, 0, 0, w, h);
    octx.restore();

    drawSpecificPoints(octx, lm, w, h, FEATURE_POINT_INDEXES, { mirror: true, radius: 1.5 });

    setBusy(true);
    try {
      const blob = await new Promise(res => off.toBlob(res, "image/jpeg", 0.9));
      const pred = await predictFaceFrame(blob);
      const savedRec = await uploadFaceFrame(blob, {
        predLabel: Number(pred?.pred_label),
        features,
      });
      setSaved(savedRec);
    } catch (e) {
      setErr("업로드/예측 실패");
    } finally {
      setBusy(false);
    }
  };

  // ---- 결과 표시 전용 유틸 (UI 폴백 포함) ----
  const renderResultBlock = () => {
    if (!saved) {
      return <p className="text-gray-500">아직 저장된 결과가 없습니다.</p>;
    }

    // 1순위: 서버가 내려준 설명 포함 결과
    const rich = saved.result_text && String(saved.result_text).trim().length > 0
      ? String(saved.result_text)
      : null;

    // 2순위: 간단 라벨/숫자로 구성된 폴백(정상/비정상)
    // 서버 필드가 케이스마다 다를 수 있어 최대한 포괄적으로 커버
    const shortLabel =
      saved.pred_label ?? saved.label ?? saved.result_label ?? saved.result ?? null;

    const shortText =
      typeof shortLabel !== "undefined" && shortLabel !== null
        ? (Number(shortLabel) === 1 ? "비정상" : (Number(shortLabel) === 0 ? "정상" : String(shortLabel)))
        : null;

    // 최종 표시 문자열
    const display = rich ?? shortText ?? "결과 텍스트가 없습니다.";

    return (
      <>
        {/* 결과 본문 */}
        <p className="text-base leading-relaxed whitespace-pre-line">
          {display}
        </p>

        {/* 메타라인 */}
        <p className="text-xs text-gray-500 mt-2">
          {typeof saved.face_id !== "undefined" && saved.face_id !== null ? `face_id: ${saved.face_id}` : ""}
          {saved.created_at ? ` · ${new Date(saved.created_at).toLocaleString()}` : ""}
        </p>
      </>
    );
  };

  return (
    <div className="w-full min-h-screen flex flex-col items-center px-6 py-10">
      <h1 className="text-3xl font-extrabold mb-6">F.A.S.T — Face 측정</h1>

      <div className="grid md:grid-cols-2 gap-8 w-full max-w-6xl">
        <div className="space-y-4">
          <div className="relative w-full rounded-2xl overflow-hidden shadow">
            <video
              ref={videoRef}
              className="w-full block"
              style={{ transform: "scaleX(-1)" }}
              muted
              playsInline
            />
            <canvas ref={overlayRef} className="absolute inset-0 w-full h-full" />
          </div>

          <button
            onClick={onCapture}
            disabled={busy || !alignOK}
            className={`px-6 py-3 rounded-2xl text-white font-semibold disabled:opacity-60 ${
              alignOK ? "bg-emerald-600 hover:bg-emerald-700" : "bg-blue-600 hover:bg-blue-700"
            }`}
          >
            {busy ? "처리 중..." : (alignOK ? "정면 OK — 측정/저장" : "측정/저장")}
          </button>
          {err && <p className="text-red-600">{err}</p>}
        </div>

        <div className="space-y-6">
          <section className="p-4 rounded-2xl border shadow-sm">
            <h2 className="text-xl font-bold mb-2">저장 (DB)</h2>
            {renderResultBlock()}
          </section>
        </div>
      </div>
    </div>
  );
}
