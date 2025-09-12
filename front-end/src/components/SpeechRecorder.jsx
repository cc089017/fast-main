// SpeechRecorder.jsx - 김민규 작성


import React, { useRef, useState } from 'react';

export default function SpeechRecorder({ onRecorded }) {
  const [recording, setRecording] = useState(false);
  const [audioUrl, setAudioUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const mediaRecorderRef = useRef(null);
  const chunks = useRef([]);
  const RECORD_DURATION = 15000; // 15초

  const startRecording = async () => {
    setError(null);
    setResult(null);
    setAudioUrl(null);
    if (recording) return;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const recorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
      chunks.current = [];
      recorder.ondataavailable = e => { if (e.data.size > 0) chunks.current.push(e.data); };
      recorder.onstop = async () => {
        const blob = new Blob(chunks.current, { type: 'audio/webm' });
        setAudioUrl(URL.createObjectURL(blob));
        setLoading(true);
        try {
          const formData = new FormData();
          formData.append('file', blob, 'recorded.webm');
          const res = await fetch('/api/v1/endpoints/speech/predict', {
            method: 'POST',
            body: formData,
          });
          const data = await res.json();
          if (!res.ok) {
            setError(data.detail || "알 수 없는 오류가 발생했습니다.");
          } else {
            setResult(data);
            if (onRecorded) onRecorded(blob, data);
          }
          setLoading(false);
        } catch (err) {
          setError("서버 연결 오류: " + err.message);
          setLoading(false);
        }
        stream.getTracks().forEach(track => track.stop());
      };
      mediaRecorderRef.current = recorder;
      recorder.start();
      setRecording(true);
      setTimeout(() => {
        if (recorder.state === 'recording') {
          recorder.stop();
          setRecording(false);
        }
      }, RECORD_DURATION);
    } catch (err) {
      console.error('마이크 권한 에러:', err);
      alert('마이크 권한을 허용해주세요. 또는 브라우저를 확인하세요.');
      setRecording(false);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && recording) {
      mediaRecorderRef.current.stop();
      setRecording(false);
    }
  };

  return (
    <div>
      {!recording ? (
        <button onClick={startRecording}>녹음 시작</button>
      ) : (
        <button onClick={stopRecording}>녹음 종료</button>
      )}
      {audioUrl && (
        <audio src={audioUrl} controls style={{ marginTop: 16 }} />
      )}
      {loading && <div style={{ marginTop: 16 }}>분석 중입니다...</div>}
      {error && <div style={{ color: 'red', marginTop: 16 }}>{error}</div>}
      {result && (
        <div style={{ marginTop: 16 }}>
          <h3>분석 결과</h3>
          {/* 실제 결과 시각화는 별도 컴포넌트로 분리 가능 */}
          <pre>{JSON.stringify(result, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}