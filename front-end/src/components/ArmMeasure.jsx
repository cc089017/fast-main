import React, { useEffect, useRef, useState } from "react";
import { postArmPredict } from "@/api/arm";

export default function ArmMeasure() {
  const videoRef = useRef(null);
  const captureCanvasRef = useRef(null);
  const leftBoxRef = useRef(null);
  const rightBoxRef = useRef(null);

  // Tasks-Vision HandLandmarker
  const landmarkerRef = useRef(null);
  const rafRef = useRef(0);

  // 자동 시작 제어
  const guideStartRef = useRef(null);   // 박스 안에서 유지 시작 시각
  const startedRef = useRef(false);     // run() 중복 방지
  const cooldownRef = useRef(0);        // 자동 재시작 쿨다운(ms)

  const b025UrlRef = useRef(null);
  const b105UrlRef = useRef(null);

  const [log, setLog] = useState([]);
  const [img025, setImg025] = useState(null);
  const [img105, setImg105] = useState(null);
  const [resp, setResp] = useState(null);
  const [loading, setLoading] = useState(false);

  const pushLog = (m) =>
    setLog((prev) => [...prev, `[${new Date().toLocaleTimeString()}] ${m}`]);

  // 비디오 준비 대기
  const waitVideoReady = () =>
    new Promise((resolve) => {
      const v = videoRef.current;
      if (!v) return resolve();
      if (v.readyState >= 2) return resolve(); // HAVE_CURRENT_DATA
      const onReady = () => {
        v.removeEventListener("loadeddata", onReady);
        resolve();
      };
      v.addEventListener("loadeddata", onReady);
    });

  // 프레임 캡처 (미러링 상태로 저장)
  const captureFrame = () => {
    const v = videoRef.current;
    const c = captureCanvasRef.current;
    const w = v.videoWidth || 1280;
    const h = v.videoHeight || 720;
    c.width = w;
    c.height = h;
    const ctx = c.getContext("2d");
    ctx.save();
    ctx.translate(w, 0);
    ctx.scale(-1, 1); // 화면과 동일하게 좌우 반전
    ctx.drawImage(v, 0, 0, w, h);
    ctx.restore();
    return new Promise((resolve) => c.toBlob((b) => resolve(b), "image/png"));
  };

  // 측정 루틴 (2.5s → 10.5s)
  const run = async () => {
    if (startedRef.current) return; // 중복 방지
    startedRef.current = true;
    setResp(null);
    setImg025(null);
    setImg105(null);
    // 기존 미리보기 URL 정리
    if (b025UrlRef.current) URL.revokeObjectURL(b025UrlRef.current);
    if (b105UrlRef.current) URL.revokeObjectURL(b105UrlRef.current);
    b025UrlRef.current = null;
    b105UrlRef.current = null;

    setLoading(true);
    pushLog("측정 시작");

    try {
      pushLog("2.5초 대기…");
      await new Promise((r) => setTimeout(r, 2500));
      const b025 = await captureFrame();
      b025UrlRef.current = URL.createObjectURL(b025);
      setImg025(b025UrlRef.current);
      pushLog("2.5s 캡처 완료");

      pushLog("10.5초(추가 8초) 대기…");
      await new Promise((r) => setTimeout(r, 8000));
      const b105 = await captureFrame();
      b105UrlRef.current = URL.createObjectURL(b105);
      setImg105(b105UrlRef.current);
      pushLog("10.5s 캡처 완료");

      const data = await postArmPredict(b025, b105);
      setResp(data);
      pushLog("서버 응답 수신");
    } catch (e) {
      const msg = e?.message || String(e);
      pushLog("오류: " + msg);
      alert(msg);
    } finally {
      setLoading(false);
      cooldownRef.current = Date.now() + 3000; // 3초 재시작 쿨다운
      startedRef.current = false;
      guideStartRef.current = null;
    }
  };

  // Tasks-Vision 결과 처리 → “좌/우 박스 모두 3초 유지 시 run()”
  const onResults = (results) => {
    const video = videoRef.current;
    const leftBox = leftBoxRef.current;
    const rightBox = rightBoxRef.current;
    if (!video || !leftBox || !rightBox) return;

    const videoRect = video.getBoundingClientRect();
    const leftRect = leftBox.getBoundingClientRect();
    const rightRect = rightBox.getBoundingClientRect();

    let inLeft = false,
      inRight = false;

    // 결과 구조 호환 처리
    const lmSets = results?.handLandmarks || results?.landmarks || [];
    const handed = results?.handednesses || results?.handedness || [];

    lmSets.forEach((lm, idx) => {
      const tip = lm[12]; // 가운데 손가락 tip
      const label =
        handed?.[idx]?.[0]?.categoryName ??
        (tip.x < 0.5 ? "Left" : "Right"); // fallback

      // 비디오를 CSS로 미러링 중이므로 x 좌표 반전 필요
      const MIRRORED = true;
      const xNorm = MIRRORED ? 1 - tip.x : tip.x;
      const tipX = videoRect.left + xNorm * videoRect.width;
      const tipY = videoRect.top + tip.y * videoRect.height;

      if (
        label === "Left" &&
        tipX >= leftRect.left &&
        tipX <= leftRect.right &&
        tipY >= leftRect.top &&
        tipY <= leftRect.bottom
      ) {
        inLeft = true;
      }
      if (
        label === "Right" &&
        tipX >= rightRect.left &&
        tipX <= rightRect.right &&
        tipY >= rightRect.top &&
        tipY <= rightRect.bottom
      ) {
        inRight = true;
      }
    });

    const bothInside = inLeft && inRight;

    if (!startedRef.current && bothInside && Date.now() > cooldownRef.current) {
      if (!guideStartRef.current) {
        guideStartRef.current = performance.now();
      } else if (performance.now() - guideStartRef.current > 3000) {
        pushLog("3초 유지 완료 — 자동 측정 시작");
        run();
      }
    } else {
      guideStartRef.current = null;
    }
  };

  // 초기화: 카메라 + Tasks-Vision HandLandmarker (CDN 동적 import → WASM 충돌 회피)
  useEffect(() => {
    let running = true;

    (async () => {
      // 카메라
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "user",
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: false,
      });
      if (videoRef.current) videoRef.current.srcObject = stream;
      await waitVideoReady();

      // Tasks-Vision 동적 import
      const vision = await import(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14"
      );
      const { FilesetResolver, HandLandmarker } = vision;

      const filesetResolver = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
      );

      const landmarker = await HandLandmarker.createFromOptions(
        filesetResolver,
        {
          baseOptions: {
            modelAssetPath:
              "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
          },
          numHands: 2,
          minDetectionConfidence: 0.5,
          minTrackingConfidence: 0.5,
          runningMode: "VIDEO",
        }
      );
      landmarkerRef.current = landmarker;

      const loop = () => {
        if (!running) return;
        const v = videoRef.current;
        if (v && v.videoWidth) {
          const results = landmarker.detectForVideo(v, performance.now());
          onResults(results);
        }
        rafRef.current = requestAnimationFrame(loop);
      };
      loop();

      pushLog("핸드 추적 시작 (Tasks-Vision)");
    })();

    return () => {
      running = false;
      cancelAnimationFrame(rafRef.current);
      try {
        landmarkerRef.current?.close();
      } catch {}
      const tracks = videoRef.current?.srcObject?.getTracks?.();
      tracks?.forEach((t) => t.stop());
      // 미리보기 URL 정리
      if (b025UrlRef.current) URL.revokeObjectURL(b025UrlRef.current);
      if (b105UrlRef.current) URL.revokeObjectURL(b105UrlRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div
      style={{
        display: "grid",
        gap: 16,
        gridTemplateColumns: "1fr 1fr",
        alignItems: "start",
      }}
    >
      {/* 좌: 비디오 & 가이드 박스 */}
      <section
        style={{
          background: "#121833",
          border: "1px solid #2a3566",
          borderRadius: 12,
          padding: 12,
        }}
      >
        <h2 style={{ marginTop: 0 }}>ARM 측정 (v1)</h2>
        <div
          style={{
            position: "relative",
            width: "100%",
            maxWidth: 640,
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
              transform: "scaleX(-1)", // 화면 미러링
            }}
          />
          {/* 가이드 박스: 좌/우 */}
          <div
            id="left-box"
            ref={leftBoxRef}
            style={{
              position: "absolute",
              left: "8%",
              top: "20%",
              width: "26%",
              height: "60%",
              border: "3px solid #4ade80",
              borderRadius: 12,
              boxShadow: "0 0 12px rgba(74,222,128,0.5) inset",
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
              border: "3px solid #60a5fa",
              borderRadius: 12,
              boxShadow: "0 0 12px rgba(96,165,250,0.5) inset",
            }}
          />
        </div>

        <div style={{ display: "flex", gap: 8, marginTop: 12 }}>
          <button
            onClick={run}
            disabled={loading || startedRef.current}
            style={{ padding: "10px 14px" }}
          >
            {loading || startedRef.current ? "진행 중…" : "측정 수동 시작"}
          </button>
        </div>

        <div
          style={{
            marginTop: 10,
            fontFamily: "ui-monospace,monospace",
            fontSize: 13,
            color: "#a9b4d4",
            whiteSpace: "pre-wrap",
          }}
        >
          {log.join("\n") || "{ 로그 없음 }"}
        </div>
      </section>

      {/* 우: 응답/미리보기 */}
      <section
        style={{
          background: "#121833",
          border: "1px solid #2a3566",
          borderRadius: 12,
          padding: 12,
        }}
      >
        <h2 style={{ marginTop: 0 }}>응답</h2>
        <pre
          style={{
            margin: 0,
            whiteSpace: "pre-wrap",
            fontFamily: "ui-monospace,monospace",
            color: "#cfe3ff",
          }}
        >
          {resp ? JSON.stringify(resp, null, 2) : "{ 아직 없음 }"}
        </pre>

        {(img025 || img105) && (
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "1fr 1fr",
              gap: 12,
              marginTop: 12,
            }}
          >
            <figure style={{ margin: 0 }}>
              <figcaption>2.5s 프레임</figcaption>
              {img025 && (
                <img
                  src={img025}
                  alt="2.5s"
                  style={{ width: "100%", borderRadius: 8 }}
                />
              )}
            </figure>
            <figure style={{ margin: 0 }}>
              <figcaption>10.5s 프레임</figcaption>
              {img105 && (
                <img
                  src={img105}
                  alt="10.5s"
                  style={{ width: "100%", borderRadius: 8 }}
                />
              )}
            </figure>
          </div>
        )}
      </section>

      <canvas ref={captureCanvasRef} style={{ display: "none" }} />
    </div>
  );
}
