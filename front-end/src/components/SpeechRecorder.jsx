// SpeechRecorder.jsx - 김수민 작성 (수정: 마운트 시 자동 녹음 시작, 시작 버튼 제거)

import React, { useRef, useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";

const MAX_SECONDS = 40;

const SpeechRecorder = ({ onStop }) => {
  const mediaRecorderRef = useRef(null);
  const streamRef = useRef(null);
  const timerRef = useRef(null);
  const chunksRef = useRef([]); // onstop 클로저 이슈 방지

  const [recording, setRecording] = useState(false);
  const [seconds, setSeconds] = useState(0);
  const [mimeType, setMimeType] = useState("audio/webm");
  const navigate = useNavigate();

  // ✅ 컴포넌트가 마운트되면 자동으로 녹음 시작
  useEffect(() => {
    startRecording();
    return () => {
      clearInterval(timerRef.current);
      try { mediaRecorderRef.current?.stop(); } catch (_) {}
      streamRef.current?.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // 타이머
  useEffect(() => {
    if (recording) {
      timerRef.current = setInterval(() => {
        setSeconds((s) => {
          if (s + 1 >= MAX_SECONDS) {
            stopRecording();
            return MAX_SECONDS;
          }
          return s + 1;
        });
      }, 1000);
    } else {
      clearInterval(timerRef.current);
    }
    return () => clearInterval(timerRef.current);
  }, [recording]);

  const startRecording = async () => {
    setSeconds(0);
    chunksRef.current = [];

    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    streamRef.current = stream;

    let options = { mimeType: "audio/webm" };
    if (window.MediaRecorder?.isTypeSupported?.("audio/webm")) {
      options = { mimeType: "audio/webm" };
    } else if (window.MediaRecorder?.isTypeSupported?.("audio/wav")) {
      options = { mimeType: "audio/wav" };
    }
    setMimeType(options.mimeType);

    const mr = new MediaRecorder(stream, options);
    mediaRecorderRef.current = mr;

    mr.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) chunksRef.current.push(e.data);
    };

    mr.onstop = () => {
      // 마이크 스트림 정리
      streamRef.current?.getTracks().forEach((t) => t.stop());
      streamRef.current = null;

      const chunks = chunksRef.current;
      if (!chunks || chunks.length === 0) {
        alert("녹음된 데이터가 없습니다. 다시 시도해 주세요.");
        return;
      }
      const blob = new Blob(chunks, { type: mimeType || "audio/webm" });
      if (!blob || blob.size === 0) {
        alert("녹음된 데이터가 없습니다. 다시 시도해 주세요.");
        return;
      }

      const filename = (mimeType || "").includes("wav")
        ? "recording.wav"
        : "recording.webm";

      onStop?.(blob, filename);   // 상위로 전달
      navigate("/test/end");      // 종료 후 이동 (원하면 경로 바꿔도 됨)
      chunksRef.current = [];
    };

    mr.start();
    setRecording(true);
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && recording) {
      try { mediaRecorderRef.current.stop(); } catch (_) {}
      setRecording(false);
    }
  };

  return (
    <div>
      <div style={{ marginBottom: 8 }}>
        {recording ? (
          <span style={{ color: "red" }}>
            ● 녹음 중... {seconds} / {MAX_SECONDS}초
          </span>
        ) : (
          <span>녹음 대기</span>
        )}
      </div>

      {/* ✅ 시작 버튼 제거: 이 컴포넌트는 mount되면 자동 녹음 */}
      {recording && (
        <button
          onClick={stopRecording}
          className="mt-8 px-7 py-4 text-[22px] bg-red-600 text-white rounded-full shadow-lg font-normal font-sans
                    hover:scale-110 hover:bg-red-700 hover:font-semibold transition-transform duration-300"
          style={{ minWidth: 200 }}
        >
          녹음 종료
        </button>
)}
    </div>
  );
};

export default SpeechRecorder;
