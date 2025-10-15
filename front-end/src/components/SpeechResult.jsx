// SpeechResult.jsx - 김민규 작성
import React, { useMemo, useState } from "react";

export default function SpeechResult({ result }) {
  if (!result) return null;

  const [showDebug, setShowDebug] = useState(false);

  const summary = useMemo(() => {
    const feats = result.features || {};
    const dbg = result.debug_info || {};
    const dtw =
      Array.isArray(dbg.dtw_means) && dbg.dtw_means.length
        ? dbg.dtw_means
        : [
            feats.dtw_slice1_mean,
            feats.dtw_slice2_mean,
            feats.dtw_slice3_mean,
            feats.dtw_slice4_mean,
            feats.dtw_slice5_mean,
          ].filter((v) => typeof v === "number");
    const avg =
      dtw.length > 0
        ? Number((dtw.reduce((a, b) => a + b, 0) / dtw.length).toFixed(2))
        : null;

    return {
      decision: result.decision,
      pred: result.pred,
      risk:
        typeof result.risk === "number" ? Number(result.risk.toFixed(3)) : result.risk,
      threshold:
        typeof result.threshold === "number"
          ? Number(result.threshold.toFixed(3))
          : result.threshold,
      dtw_means: dtw.map((v) => Number(v.toFixed?.(2) ?? v)),
      dtw_avg: avg,
      dtw_slope:
        typeof (dbg.dtw_slope ?? feats.dtw_mean_slope) === "number"
          ? Number((dbg.dtw_slope ?? feats.dtw_mean_slope).toFixed(5))
          : dbg.dtw_slope ?? feats.dtw_mean_slope,
      audio_duration:
        typeof dbg.audio_duration === "number"
          ? Number(dbg.audio_duration.toFixed(2))
          : null,
      filename: dbg.filename || null,
    };
  }, [result]);

  const decisionColor = result.decision === "Abnormal" ? "text-red-600" : "text-green-600";
  const decisionBg = result.decision === "Abnormal" ? "bg-red-50" : "bg-green-50";
  const riskPct =
    typeof result.risk === "number" ? Math.round(result.risk * 100) : result.risk;

  return (
    <div className="mt-6 space-y-4">
      {/* 그래프 최상단 배치 */}
      {result.graph && (
        <div>
          <h3 className="text-lg font-semibold mb-2">시각화</h3>
          <img
            src={result.graph}
            alt="Speech analysis graph"
            className="w-full rounded border shadow"
          />
        </div>
      )}

      {/* 핵심 결과 카드 */}
      <div className={`p-4 rounded border ${decisionBg}`}>
        <div className="flex flex-wrap items-center gap-3">
          <span className="text-sm text-gray-600">판정</span>
          <span className={`text-xl font-bold ${decisionColor}`}>
            {result.decision}
          </span>
          <span className="ml-2 text-sm text-gray-600">
            위험도 {riskPct}% · 임계값{" "}
            {typeof result.threshold === "number"
              ? Math.round(result.threshold * 100)
              : result.threshold}
            %
          </span>
          {summary.filename && (
            <span className="ml-auto text-xs text-gray-500">파일: {summary.filename}</span>
          )}
        </div>
      </div>

      {/* 요약 JSON (중요 정보만) */}
      <div>
        <h4 className="text-md font-semibold mb-1">요약</h4>
        <pre className="text-sm bg-gray-50 rounded border p-3 overflow-x-auto">
{JSON.stringify(summary, null, 2)}
        </pre>
      </div>

      {/* 전체 디버그 JSON: 기본 접힘 */}
      <div>
        <button
          onClick={() => setShowDebug((s) => !s)}
          className="text-sm text-blue-600 hover:underline"
        >
          {showDebug ? "디버그 숨기기" : "디버그 상세 보기"}
        </button>
        {showDebug && (
          <pre className="mt-2 text-xs bg-gray-50 rounded border p-3 overflow-auto max-h-[300px]">
{JSON.stringify(result, null, 2)}
          </pre>
        )}
      </div>
    </div>
  );
}