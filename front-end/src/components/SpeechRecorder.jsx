// SpeechRecorder.jsx - 김민규 작성

import React, { useRef, useState } from 'react';

// 안정적인 녹음 기능 구현, MediaRecorder는 useRef로 관리
export default function SpeechRecorder({ onRecorded }) {
  const [recording, setRecording] = useState(false);
  const mediaRecorderRef = useRef(null);
  const chunks = useRef([]);

  const startRecording = async () => {
    if (recording) return;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      recorder.ondataavailable = e => chunks.current.push(e.data);
      recorder.onstop = () => {
        const blob = new Blob(chunks.current, { type: 'audio/webm' });
        onRecorded(blob);
        chunks.current = [];
        stream.getTracks().forEach(track => track.stop()); // 마이크 해제
      };
      mediaRecorderRef.current = recorder;
      recorder.start();
      setRecording(true);
      setTimeout(() => {
        recorder.stop();
        setRecording(false);
      }, 12000); // 12초 녹음
    } catch  {
      alert('마이크 권한을 허용해주세요. 또는 브라우저를 확인하세요.');
      setRecording(false);
    }
  };

  return (
    <div>
      <button onClick={startRecording} disabled={recording}>
        {recording ? '녹음 중...' : '녹음 시작'}
      </button>
    </div>
  );
}