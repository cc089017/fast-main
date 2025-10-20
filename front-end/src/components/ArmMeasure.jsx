import React, { useEffect, useRef, useState } from "react";
import { postArmPredict } from "@/api/arm";
import { useNavigate } from "react-router-dom";

export default function ArmMeasure() {
  const videoRef = useRef(null);
  const captureCanvasRef = useRef(null);
  const leftBoxRef = useRef(null);
  const rightBoxRef = useRef(null);

  const handsRef = useRef(null);
  const cameraRef = useRef(null);

  const guideStartRef = useRef(null);
  const startedRef = useRef(false);
  const cooldownRef = useRef(0);

  const navigate = useNavigate();

  // 안내 멘트 상태
  // idle → hold(3초 유지) → running(검사 진행) → done(종료)
  const [status, setStatus] = useState("idle");
  const statusText = {
    idle: "손이 하늘로 향하게 한채로 박스안에 손이 보이게 넣어주세요",
    hold: "잠시후 검사가 시작됩니다. 눈을 감아주세요",
    running: "검사 진행중",
    done: "검사가 종료되었습니다",
  };

  const waitVideoReady = () =>
    new Promise((resolve) => {
      const v = videoRef.current;
      if (!v) return resolve();
      if (v.readyState >= 2) return resolve();
      const onReady = () => {
        v.removeEventListener("loadeddata", onReady);
        resolve();
      };
      v.addEventListener("loadeddata", onReady);
    });

  // 프레임 캡처(화면과 동일하게 좌우 반전)
  const captureFrame = () =>
    new Promise((resolve, reject) => {
      const v = videoRef.current;
      const c = captureCanvasRef.current;
      const w = v?.videoWidth || 1280;
      const h = v?.videoHeight || 720;
      c.width = w;
      c.height = h;
      const ctx = c.getContext("2d");
      ctx.save();
      ctx.translate(w, 0);
      ctx.scale(-1, 1);
      ctx.drawImage(v, 0, 0, w, h);
      ctx.restore();
      c.toBlob((b) => (b ? resolve(b) : reject(new Error("캡처 실패"))), "image/png");
    });

  // 2.5s → 10.5s 측정 루틴
  const run = async () => {
    if (startedRef.current) return;
    startedRef.current = true;
    setStatus("running");

    try {
      await new Promise((r) => setTimeout(r, 2500));
      const b025 = await captureFrame();

      await new Promise((r) => setTimeout(r, 8000));
      const b105 = await captureFrame();

      await postArmPredict(b025, b105); // 서버 업로드(백엔드가 DB 저장)

      setStatus("done");
      setTimeout(() => navigate("/test/speech", { replace: true }), 1000);
    } catch (e) {
      const msg = e?.response?.data?.detail || e?.message || String(e);
      alert(msg);
      setStatus("idle");
    } finally {
      cooldownRef.current = Date.now() + 3000; // 3초 쿨다운
      startedRef.current = false;
      guideStartRef.current = null;
    }
  };

  // Hands 결과 처리 → 양손이 각 박스에 3초 유지되면 run()
  const onResults = (results) => {
    const video = videoRef.current;
    const leftBox = leftBoxRef.current;
    const rightBox = rightBoxRef.current;
    if (!video || !leftBox || !rightBox) return;

    const videoRect = video.getBoundingClientRect();
    const leftRect = leftBox.getBoundingClientRect();
    const rightRect = rightBox.getBoundingClientRect();

    let inLeft = false;
    let inRight = false;

    const lmSets = results?.multiHandLandmarks || [];
    lmSets.forEach((lm) => {
      const tip = lm?.[12]; // middle finger tip
      if (!tip) return;
      const x = videoRect.left + tip.x * videoRect.width;
      const y = videoRect.top + tip.y * videoRect.height;
      if (x >= leftRect.left && x <= leftRect.right && y >= leftRect.top && y <= leftRect.bottom)
        inLeft = true;
      if (x >= rightRect.left && x <= rightRect.right && y >= rightRect.top && y <= rightRect.bottom)
        inRight = true;
    });

    const bothInside = inLeft && inRight;

    if (!startedRef.current && bothInside && Date.now() > cooldownRef.current) {
      if (!guideStartRef.current) {
        guideStartRef.current = performance.now();
        setStatus("hold");
      } else if (performance.now() - guideStartRef.current > 3000) {
        setStatus("running");
        run();
      }
    } else {
      guideStartRef.current = null;
      if (!startedRef.current) setStatus("idle");
    }
  };

  // 초기화(카메라 + MediaPipe v0.4)
  useEffect(() => {
    let running = true;

    const loadScript = (src) =>
      new Promise((resolve, reject) => {
        if ([...document.scripts].some((s) => s.src === src)) return resolve();
        const s = document.createElement("script");
        s.src = src;
        s.async = true;
        s.onload = resolve;
        s.onerror = () => reject(new Error(`Failed to load: ${src}`));
        document.head.appendChild(s);
      });

    (async () => {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      if (videoRef.current) videoRef.current.srcObject = stream;
      await waitVideoReady();

      const VER = "0.4";
      await loadScript(`https://cdn.jsdelivr.net/npm/@mediapipe/hands@${VER}/hands.js`);
      try {
        await loadScript("https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js");
      } catch {
        await loadScript("https://unpkg.com/@mediapipe/camera_utils/camera_utils.js");
      }

      const Hands = window.Hands;
      const Camera = window.Camera;
      if (!Hands || !Camera) throw new Error("MediaPipe Hands 또는 Camera 유틸 로드 실패");

      const hands = new Hands({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands@${VER}/${file}`,
      });
      hands.setOptions({
        selfieMode: true,
        maxNumHands: 2,
        modelComplexity: 1,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });
      hands.onResults(onResults);
      handsRef.current = hands;

      const cam = new Camera(videoRef.current, {
        onFrame: async () => {
          if (!running) return;
          await hands.send({ image: videoRef.current });
        },
        width: 1280,
        height: 720,
      });
      cam.start();
      cameraRef.current = cam;
    })().catch((err) => {
      alert("초기화 오류: " + (err?.response?.data?.detail || err?.message || String(err)));
    });

    return () => {
      running = false;
      try { handsRef.current?.close?.(); } catch {}
      try { cameraRef.current?.stop?.(); } catch {}
      const tracks = videoRef.current?.srcObject?.getTracks?.();
      tracks?.forEach((t) => t.stop());
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="w-screen h-screen flex items-center justify-center bg-white relative overflow-hidden">
      <div className="w-[133vh] h-[133vh] rounded-full border-[7vw] border-[#f6f6f6] shadow-xl overflow-hidden flex items-center justify-center z-0 relative">
        <div
          style={{
            position: "relative",
            width: "100%",
            maxWidth: 850,
            aspectRatio: "16/9",
            borderRadius: 12,
            overflow: "hidden",
          }}
        >
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            style={{
              width: "100%",
              height: "100%",
              objectFit: "cover",
              transform: "scaleX(-1)",
            }}
          />

          {/* 중앙 상단 초록 멘트 */}
          <div
            style={{
              position: "absolute",
              top: 12,
              left: "50%",
              transform: "translateX(-50%)",
              width: "90%",
              textAlign: "center",
              fontSize: 20,
              fontWeight: 800,
              color: "#22c55e",                 // 초록 글씨
              textShadow: "0 1px 2px rgba(0,0,0,0.35)",
              pointerEvents: "none",
              userSelect: "none",
            }}
          >
            {statusText[status]}
          </div>

          {/* 좌/우 가이드 박스(검정 테두리) */}
          <div
            id="left-box"
            ref={leftBoxRef}
            style={{
              position: "absolute",
              left: "8%",
              top: "20%",
              width: "26%",
              height: "60%",
              border: "3px solid #000",
              borderRadius: 12,
              boxShadow: "0 0 12px rgba(0,0,0,0.5) inset",
            }}
          />
          <div
            id="right-box"
            ref={rightBoxRef}
            style={{
              position: "absolute",
              right: "8%",
              top: "20%",
              width: "26%",
              height: "60%",
              border: "3px solid #000",
              borderRadius: 12,
              boxShadow: "0 0 12px rgba(0,0,0,0.5) inset",
            }}
          />
        </div>
      </div>

      <canvas ref={captureCanvasRef} style={{ display: "none" }} />
    </div>
  );
}
