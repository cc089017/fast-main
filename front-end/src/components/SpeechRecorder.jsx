// SpeechRecorder.jsx - 김민규 작성


import React, { useRef, useState } from 'react';

export default function SpeechRecorder({ onRecorded }) {
  const [recording, setRecording] = useState(false);
  const [timer, setTimer] = useState(0);
  const mediaRecorderRef = useRef(null);
  const chunks = useRef([]);
  const maxDuration = 40; // 녹음 제한(초)

  const startRecording = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorderRef.current = new MediaRecorder(stream, { mimeType: 'audio/webm' });
    chunks.current = [];
    mediaRecorderRef.current.ondataavailable = (e) => {
      if (e.data.size > 0) chunks.current.push(e.data);
    };
    mediaRecorderRef.current.onstop = () => {
      const blob = new Blob(chunks.current, { type: 'audio/webm' });
      onRecorded(blob);
    };
    mediaRecorderRef.current.start();
    setRecording(true);
    setTimer(0);

    // 자동 종료 타이머
    const interval = setInterval(() => {
      setTimer((t) => {
        if (t + 1 >= maxDuration) {
          stopRecording();
          clearInterval(interval);
        }
        return t + 1;
      });
    }, 1000);
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && recording) {
      mediaRecorderRef.current.stop();
      setRecording(false);
    }
  };

  return (
    <div>
      <button onClick={recording ? stopRecording : startRecording}>
        {recording ? '녹음 종료' : '녹음 시작'}
      </button>
      {recording && <p>녹음 중... {timer}초 / {maxDuration}초</p>}
    </div>
  );
}