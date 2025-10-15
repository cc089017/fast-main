import React, { useState, useRef } from "react";
import SpeechRecorder from "./SpeechRecorder";
import SpeechResult from "./SpeechResult";

const promptText = `
아래 문장을 읽어주세요:

안녕하세요 제 이름은 000입니다.
저는 000년 00월 00일에 00에서 태어났고 00살입니다.
00초등학교 00중학교 00고등학교 00대학교를 졸업했고
지금 000일을 하고 있습니다. 감사합니다
`;

export default function SpeechTestPage() {
  const [result, setResult] = useState(null);
  const [recording, setRecording] = useState(false);
  const [loading, setLoading] = useState(false);
  const resultRef = useRef(null);

  // 추가: 버튼 핸들러(간단 스텁, 기존 UI 유지용)
  const checkModelInfo = () => {
    console.log("[INFO] checkModelInfo clicked");
    alert("모델 정보 확인은 준비 중입니다.");
  };
  const checkSystem = () => {
    console.log("[INFO] checkSystem clicked");
    alert("시스템 환경 확인은 준비 중입니다.");
  };

  // 녹음 종료 시 서버로 전송
  const handleStop = async (blob, filename = "recording.webm") => {
    console.log("[DEBUG] handleStop called with blob:", blob.size, "bytes");

    setLoading(true);
    setResult(null);
    setRecording(false); // 녹음 상태 종료

    try {
      const fd = new FormData();
      fd.append("file", blob, filename);

      console.log("[DEBUG] Sending to API:", filename, blob.size, "bytes");

      const res = await fetch("http://127.0.0.1:8000/api/v1/speech/predict", {
        method: "POST",
        body: fd,
      });

      console.log("Response status:", res.status);

      if (res.ok) {
        const data = await res.json();
        console.log("API 성공 응답:", data);
        setResult(data);
        // 결과 영역으로 스크롤
        setTimeout(() => resultRef.current?.scrollIntoView({ behavior: "smooth" }), 50);
      } else {
        const errorText = await res.text();
        console.error("API Error:", res.status, errorText);
        setResult({
          error: `API Error: ${res.status} - ${errorText}`,
        });
      }
    } catch (e) {
      console.error("Request Error:", e);
      setResult({
        error: `Request Error: ${e.message}`,
      });
    }

    setLoading(false);
  };

  // 업로드 완료 결과
  const handleDone = (data) => {
    setResult(data);
    setTimeout(() => resultRef.current?.scrollIntoView({ behavior: "smooth" }), 50);
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
            <div className="flex flex-col gap-4 mb-8 items-center">
              <button
                onClick={checkModelInfo}
                className="px-6 py-2 bg-purple-600 text-white rounded hover:bg-purple-700"
              >
                모델 정보 확인
              </button>
              <button
                onClick={checkSystem}
                className="px-6 py-2 bg-orange-600 text-white rounded hover:bg-orange-700"
              >
                시스템 환경 확인
              </button>
              <button
                className="mt-8 px-7 py-4 text-4xl bg-blue-600 text-white rounded-full shadow-lg font-normal font-sans
                         hover:scale-110 hover:bg-blue-700 hover:font-semibold transition-transform duration-300"
                style={{ minWidth: 200 }}
                onClick={() => setRecording(true)}
              >
                녹음 시작 준비
              </button>
            </div>
          )}

          {/* 녹음 컴포넌트 */}
          {recording && <SpeechRecorder onStop={handleStop} onDone={handleDone} />}

          {/* 결과 */}
          <div ref={resultRef}>{result ? <SpeechResult result={result} /> : null}</div>
        </div>
      </div>
    </div>
  );
};
