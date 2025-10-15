// SpeechRecorder.jsx - 김수민 작성 (수정: 수동 시작/종료로 변경)

import React, { useRef, useState, useEffect, useCallback } from "react";

const MAX_SECONDS = 40;

const SpeechRecorder = ({ onStop, onDone }) => {
  const mediaRecorderRef = useRef(null);
  const streamRef = useRef(null);
  const timerRef = useRef(null);
  const chunksRef = useRef([]);

  const [recording, setRecording] = useState(false);
  const [seconds, setSeconds] = useState(0);
  const [mimeType, setMimeType] = useState("audio/webm");
  const [microphoneReady, setMicrophoneReady] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [fileName, setFileName] = useState("");

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && recording) {
      try { 
        mediaRecorderRef.current.stop(); 
        console.log("[DEBUG] Recording stopped manually");
      } catch (e) {
        console.error("Stop recording error:", e);
      }
      setRecording(false);
    }
  }, [recording]);

  const startRecording = useCallback(async () => {
    console.log("[DEBUG] Starting recording...");
    setSeconds(0);
    chunksRef.current = [];

    try {
      // 마이크 권한 요청
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          sampleRate: 16000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true
        } 
      });
      streamRef.current = stream;
      setMicrophoneReady(true);
      console.log("[DEBUG] Microphone access granted");

      // MIME 타입 설정
      let options = { mimeType: "audio/webm" };
      if (window.MediaRecorder?.isTypeSupported?.("audio/wav")) {
        options = { mimeType: "audio/wav" };
        console.log("[DEBUG] Using WAV format");
      } else if (window.MediaRecorder?.isTypeSupported?.("audio/webm;codecs=opus")) {
        options = { mimeType: "audio/webm;codecs=opus" };
        console.log("[DEBUG] Using WebM with Opus codec");
      } else if (window.MediaRecorder?.isTypeSupported?.("audio/webm")) {
        options = { mimeType: "audio/webm" };
        console.log("[DEBUG] Using WebM format");
      } else {
        console.log("[DEBUG] Using default format");
      }
      
      setMimeType(options.mimeType || "audio/webm");

      // MediaRecorder 생성
      const mr = new MediaRecorder(stream, options);
      mediaRecorderRef.current = mr;

      mr.ondataavailable = (e) => {
        console.log("[DEBUG] Data available:", e.data.size, "bytes");
        if (e.data && e.data.size > 0) {
          chunksRef.current.push(e.data);
        }
      };

      mr.onstop = () => {
        console.log("[DEBUG] MediaRecorder stopped");
        
        // 스트림 정리
        streamRef.current?.getTracks().forEach((t) => t.stop());
        streamRef.current = null;
        setMicrophoneReady(false);

        const chunks = chunksRef.current;
        console.log("[DEBUG] Total chunks:", chunks.length);
        
        if (!chunks || chunks.length === 0) {
          alert("녹음된 데이터가 없습니다. 다시 시도해 주세요.");
          return;
        }
        
        const blob = new Blob(chunks, { type: mimeType || "audio/webm" });
        console.log("[DEBUG] Blob created:", blob.size, "bytes");
        
        if (!blob || blob.size === 0) {
          alert("녹음된 데이터가 없습니다. 다시 시도해 주세요.");
          return;
        }

        const filename = (mimeType || "").includes("wav") ? "recording.wav" : "recording.webm";
        
        console.log("[DEBUG] Calling onStop with blob:", blob.size, "bytes");
        onStop?.(blob, filename);
        chunksRef.current = [];
      };

      mr.onerror = (e) => {
        console.error("[ERROR] MediaRecorder error:", e);
        alert("녹음 중 오류가 발생했습니다: " + e.error);
      };

      // 녹음 시작
      mr.start(1000); // 1초마다 데이터 수집
      setRecording(true);
      console.log("[DEBUG] Recording started");

    } catch (error) {
      console.error("녹음 시작 실패:", error);
      setMicrophoneReady(false);
      if (error.name === 'NotAllowedError') {
        alert("마이크 접근 권한이 필요합니다. 브라우저 설정을 확인해주세요.");
      } else {
        alert("마이크에 접근할 수 없습니다: " + error.message);
      }
    }
  }, [mimeType, onStop]);

  // 컴포넌트가 마운트될 때 자동으로 녹음 시작하지 않음
  useEffect(() => {
    return () => {
      // 컴포넌트 언마운트 시 정리
      clearInterval(timerRef.current);
      try { 
        mediaRecorderRef.current?.stop(); 
      } catch (_) {
        // 무시
      }
      streamRef.current?.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    };
  }, []);

  // 타이머 관리
  useEffect(() => {
    if (recording) {
      timerRef.current = setInterval(() => {
        setSeconds((s) => {
          const newSeconds = s + 1;
          if (newSeconds >= MAX_SECONDS) {
            console.log("[DEBUG] Max time reached, stopping recording");
            stopRecording();
            return MAX_SECONDS;
          }
          return newSeconds;
        });
      }, 1000);
    } else {
      clearInterval(timerRef.current);
    }
    return () => clearInterval(timerRef.current);
  }, [recording, stopRecording]);

  const onFileChange = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setFileName(file.name);
    try {
      setUploading(true);
      const fd = new FormData();
      fd.append("file", file, file.name);
      const res = await fetch("http://127.0.0.1:8000/api/v1/speech/predict", {
        method: "POST",
        body: fd,
      });
      const data = await res.json();
      onDone?.(data); // 결과를 부모(SpeechTestPage)로 전달
      console.debug("[UPLOAD] result:", data);
    } catch (err) {
      console.error("[UPLOAD] failed:", err);
      alert("파일 업로드에 실패했습니다.");
    } finally {
      setUploading(false);
      e.target.value = "";
    }
  };

  return (
    <div className="space-y-4">
      <div style={{ marginBottom: 16 }} className="text-center">
        {!microphoneReady && !recording ? (
          <span className="text-gray-600">마이크 준비 중...</span>
        ) : recording ? (
          <span style={{ color: "red", fontSize: "18px", fontWeight: "bold" }}>
            🔴 녹음 중... {seconds} / {MAX_SECONDS}초
          </span>
        ) : (
          <span className="text-green-600">녹음 준비 완료</span>
        )}
      </div>

      {!recording ? (
        <button
          onClick={startRecording}
          className="px-8 py-4 text-2xl bg-red-600 text-white rounded-full shadow-lg font-semibold
                     hover:scale-110 hover:bg-red-700 transition-transform duration-300"
          style={{ minWidth: 200 }}
        >
          🎤 녹음 시작
        </button>
      ) : (
        <button
          onClick={stopRecording}
          className="px-8 py-4 text-2xl bg-gray-600 text-white rounded-full shadow-lg font-semibold
                     hover:scale-110 hover:bg-gray-700 transition-transform duration-300"
          style={{ minWidth: 200 }}
        >
          ⏹️ 녹음 종료
        </button>
      )}

      {recording && (
        <div className="mt-4 text-center">
          <p className="text-sm text-gray-600">
            최소 5-10초 이상 명확하게 말씀해주세요
          </p>
          <p className="text-xs text-gray-500">
            {MAX_SECONDS}초 후 자동 종료됩니다
          </p>
        </div>
      )}

      {/* 파일 업로드로 검사 (추가) */}
      <div className="mt-6 p-4 border rounded">
        <h3 className="font-semibold mb-2">파일 업로드로 검사</h3>
        <p className="text-sm text-gray-500 mb-2">
          WAV/MP3/M4A/OGG/WebM 등 지원 (최대 30MB 권장)
        </p>
        <label className="inline-block">
          <input
            type="file"
            accept="audio/*,.wav,.mp3,.m4a,.ogg,.webm"
            onChange={onFileChange}
            disabled={uploading}
            className="hidden"
          />
          <span className="px-4 py-2 bg-blue-600 text-white rounded cursor-pointer">
            {uploading ? "업로드 중..." : "파일 선택"}
          </span>
        </label>
        <span className="ml-3 text-sm text-gray-600">
          {fileName || "선택된 파일 없음"}
        </span>
      </div>
    </div>
  );
};

export default SpeechRecorder;
