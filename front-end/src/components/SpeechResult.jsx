// SpeechResult.jsx - 김민규 작성
import React from 'react';

export default function SpeechResult({ result, plotUrl }) {
  return (
    <div>
      <h2>음성 검사 결과</h2>
      <img src={plotUrl} alt="검사 결과 그래프" style={{ maxWidth: '100%' }} />
      <div>
        <p>판정: {result.decision}</p>
        <p>위험도: {result.risk}</p>
        <p>임계값: {result.threshold}</p>
        {/* 특징 벡터 등 추가 정보 표시 */}
      </div>
    </div>
  );
}