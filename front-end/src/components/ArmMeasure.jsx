import React, { useEffect, useRef, useState } from "react";
import { postArmPredict } from "@/api/arm";
import { useNavigate } from "react-router-dom";

export default function ArmMeasure() {
  // ====== 픽셀 고정 레이아웃 상수 ======
  const VIDEO_W = 870;
  const VIDEO_H = 450;

  // 박스 픽셀 고정 (좌/우 동일 크기)
  const BOX_W = 250;
  const BOX_H = 320;

  // 박스 위치(비디오 좌측상단 기준 픽셀 고정)
  const LEFT_BOX = { x: 80, y: 110 };
  const RIGHT_BOX = { x: VIDEO_W - 80 - BOX_W, y: 110 };

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

  // 화면과 동일하게 좌우 반전하여 캡처
  const captureFrame = () =>
    new Promise((resolve, reject) => {
      const v = videoRef.current;
      const c = captureCanvasRef.current;
      const w = v?.videoWidth || VIDEO_W;
      const h = v?.videoHeight || VIDEO_H;
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

  const run = async () => {
    if (startedRef.current) return;
    startedRef.current = true;
    setStatus("running");

    try {
      await new Promise((r) => setTimeout(r, 2500));
      const b025 = await captureFrame();

      await new Promise((r) => setTimeout(r, 8000));
      const b105 = await captureFrame();

      await postArmPredict(b025, b105);

      setStatus("done");
      setTimeout(() => navigate("/test/speech", { replace: true }), 1000);
    } catch (e) {
      const msg = e?.response?.data?.detail || e?.message || String(e);
      alert(msg);
      setStatus("idle");
    } finally {
      cooldownRef.current = Date.now() + 3000;
      startedRef.current = false;
      guideStartRef.current = null;
    }
  };

  // Hands 결과 처리: 고정 크기/좌표로 판정
  const onResults = (results) => {
    const video = videoRef.current;
    const leftBox = leftBoxRef.current;
    const rightBox = rightBoxRef.current;
    if (!video || !leftBox || !rightBox) return;

    // 비디오의 화면 내 위치(고정 픽셀로 잡았지만, 센터 정렬로 인한 오프셋 고려)
    const videoRect = video.getBoundingClientRect();
    const baseLeft = videoRect.left;
    const baseTop = videoRect.top;

    let inLeft = false;
    let inRight = false;

    const lmSets = results?.multiHandLandmarks || [];
    lmSets.forEach((lm) => {
      const tip = lm?.[12]; // middle finger tip
      if (!tip) return;

      // MediaPipe 좌표(0~1)를 비디오 픽셀 좌표로 (표시 크기 고정)
      const px = baseLeft + tip.x * VIDEO_W;
      const py = baseTop + tip.y * VIDEO_H;

      const L = {
        left: baseLeft + LEFT_BOX.x,
        right: baseLeft + LEFT_BOX.x + BOX_W,
        top: baseTop + LEFT_BOX.y,
        bottom: baseTop + LEFT_BOX.y + BOX_H,
      };
      const R = {
        left: baseLeft + RIGHT_BOX.x,
        right: baseLeft + RIGHT_BOX.x + BOX_W,
        top: baseTop + RIGHT_BOX.y,
        bottom: baseTop + RIGHT_BOX.y + BOX_H,
      };

      if (px >= L.left && px <= L.right && py >= L.top && py <= L.bottom) inLeft = true;
      if (px >= R.left && px <= R.right && py >= R.top && py <= R.bottom) inRight = true;
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
  }, []); // eslint-disable-line

  return (
      <div className="w-screen h-screen flex items-center justify-center bg-white relative overflow-hidden">
      {/* ✅ 겉 원: 원래 코드 유지 */}
        <div className="w-[133vh] h-[133vh] rounded-full border-[7vw] border-[#f6f6f6] shadow-xl overflow-hidden flex items-center justify-center z-0 relative">
          {/* 🔒 안쪽 카메라: 픽셀 고정 */}
          
        {/* 카메라 고정 박스 */}
        <div
          style={{
            position: "relative",
            width: VIDEO_W,
            height: VIDEO_H,
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
              width: VIDEO_W,
              height: VIDEO_H,
              objectFit: "cover",
              transform: "scaleX(-1)",
              display: "block",
            }}
          />

          {/* 중앙 상단 안내 멘트 (고정 px) */}
          <div
            style={{
              position: "absolute",
              top: 12,
              left: 0,
              width: VIDEO_W,
              textAlign: "center",
              fontSize: 20,
              fontWeight: 800,
              color: "#22c55e",
              textShadow: "0 1px 2px rgba(0,0,0,0.35)",
              pointerEvents: "none",
              userSelect: "none",
            }}
          >
            {statusText[status]}
          </div>

          {/* 좌/우 가이드 박스(픽셀 고정) */}
          <div
            id="left-box"
            ref={leftBoxRef}
            style={{
              position: "absolute",
              left: LEFT_BOX.x,
              top: LEFT_BOX.y,
              width: BOX_W,
              height: BOX_H,
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
              left: RIGHT_BOX.x,
              top: RIGHT_BOX.y,
              width: BOX_W,
              height: BOX_H,
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
