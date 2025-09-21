// SpeechRecorder.jsx - 김민규 작성


import React, { useRef, useState, useEffect } from "react";

const MAX_SECONDS = 40;

const SpeechRecorder = ({ onStop }) => {
  const mediaRecorderRef = useRef(null);
  const [recording, setRecording] = useState(false);
  const [chunks, setChunks] = useState([]);
  const [seconds, setSeconds] = useState(0);
  const timerRef = useRef(null);

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
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    let options = { mimeType: "audio/webm" };
    if (MediaRecorder.isTypeSupported("audio/webm")) {
      options = { mimeType: "audio/webm" };
    } else if (MediaRecorder.isTypeSupported("audio/wav")) {
      options = { mimeType: "audio/wav" };
    }
    mediaRecorderRef.current = new MediaRecorder(stream, options);
    setChunks([]);
    mediaRecorderRef.current.ondataavailable = (e) => {
      setChunks((prev) => [...prev, e.data]);
    };
    mediaRecorderRef.current.onstop = () => {
      if (chunks.length === 0) {
        alert("녹음된 데이터가 없습니다. 다시 시도해 주세요.");
        return;
      }
      const blob = new Blob(chunks, { type: "audio/webm" });
      if (blob.size === 0) {
        alert("녹음된 데이터가 없습니다. 다시 시도해 주세요.");
        return;
      }
      onStop(blob, "recording.webm");
      setChunks([]);
    };
    mediaRecorderRef.current.start();
    setRecording(true);
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && recording) {
      mediaRecorderRef.current.stop();
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
      {!recording ? (
        <button onClick={startRecording}>녹음 시작</button>
      ) : (
        <button onClick={stopRecording}>녹음 종료</button>
      )}
    </div>
  );
};

export default SpeechRecorder;

