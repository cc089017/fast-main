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
        console.log("실제 API 응답:", data);
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
    <div className="w-screen h-screen flex items-center justify-center bg-white relative overflow-hidden">
      {/* 원형 링 배경 */}
      <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
        <div className="w-[133vh] h-[133vh] rounded-full border-[7vw] border-[#f6f6f6] shadow-xl overflow-hidden" />
      </div>

      {/* 콘텐츠 카드 */}
      <div className="relative z-10 w-full max-w-xl">
        <div className="bg-white rounded-xl shadow-lg p-8 flex flex-col items-center">
          <div className="mb-8 text-center">
            <h2 className="text-4xl font-bold mb-4 text-blue-700">음성 검사</h2>
            <pre className="text-gray-800 text-[20px] font-sans font-normal whitespace-pre-wrap">
              {promptText}
            </pre>
          </div>

          {loading && (
            <div className="my-8 text-3xl text-blue-600 font-bold">결과 출력중...</div>
          )}

          {!loading && !result && !recording && (
            <button
              className="mt-8 px-7 py-4 text-4xl bg-blue-600 text-white rounded-full shadow-lg font-normal font-sans
                         hover:scale-110 hover:bg-blue-700 hover:font-semibold transition-transform duration-300"
              style={{ minWidth: 200 }}
              onClick={() => setRecording(true)}
            >
              녹음 시작
            </button>
          )}

          {recording && <SpeechRecorder onStop={handleStop} />}

          {result && !loading && <SpeechResult result={result} />}
        </div>
      </div>
    </div>
  );
};

export default SpeechTestPage;
