import React, { useState } from "react";
import SpeechRecorder from "./SpeechRecorder";
import SpeechResult from "./SpeechResult";

const promptText = `
아래 문장을 읽어주세요:
안녕하세요 제 이름은 000입니다.
저는 000년 00월 00일에 00에서 태어났고 00살입니다.
00초등학교 00중학교 00고등학교 00대학교를 졸업했고
지금 000일을 하고 있습니다. 감사합니다
`;

const SpeechTestPage = () => {
  const [recording, setRecording] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  // 녹음 종료 시 호출
  const handleStop = async (blob, filename) => {
    setLoading(true);
    setResult(null);
    
    try {
      const formData = new FormData();
      formData.append("file", blob, filename);
      
      const res = await fetch("/api/v1/endpoints/speech/predict", {
        method: "POST",
        body: formData,
      });
      
      if (res.ok) {
        const data = await res.json();
        console.log("실제 API 응답:", data); // 이 로그로 실제 응답 구조 확인
        setResult(data);
      } else {
        console.error("API Error:", res.status);
        setResult({ error: "API Error" });
      }
    } catch (e) {
      console.error("Request Error:", e);
      setResult({ error: "Request Error" });
    }
    
    setLoading(false);
    setRecording(false);
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gray-50">
      <div className="bg-white rounded-xl shadow-lg p-8 flex flex-col items-center w-full max-w-xl">
        <div className="mb-8 text-center">
          <h2 className="text-2xl font-bold mb-4 text-blue-700">음성 검사</h2>
          <pre className="text-gray-800 text-lg whitespace-pre-wrap">{promptText}</pre>
        </div>
        {loading && (
          <div className="my-8 text-xl text-blue-600 font-bold">결과 출력중...</div>
        )}
        {!loading && !result && !recording && (
          <button
            className="mt-8 px-10 py-4 text-xl bg-blue-600 text-white rounded-full shadow-lg hover:bg-blue-700 transition font-semibold"
            style={{ minWidth: 200 }}
            onClick={() => setRecording(true)}
          >
            녹음 시작
          </button>
        )}
        {recording && (
          <SpeechRecorder onStop={handleStop} />
        )}
        {result && !loading && (
          <SpeechResult result={result} />
        )}
      </div>
    </div>
  );
};

export default SpeechTestPage;