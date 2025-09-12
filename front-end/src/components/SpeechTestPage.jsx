import React, { useState } from 'react';
import SpeechRecorder from './SpeechRecorder';
import SpeechResult from './SpeechResult';
import { sendSpeechAudio, getSpeechPlot, saveSpeechResult } from '../api/speech';

export default function SpeechTestPage() {
  const [audioBlob, setAudioBlob] = useState(null);
  const [result, setResult] = useState(null);
  const [plotUrl, setPlotUrl] = useState(null);

  const handleRecorded = async blob => {
    setAudioBlob(blob);
    const res = await sendSpeechAudio(blob);
    setResult(res);
    const plotBlob = await getSpeechPlot(blob);
    setPlotUrl(URL.createObjectURL(plotBlob));
    await saveSpeechResult({ ...res, audio: blob });
  };

  return (
    <div>
      {!audioBlob && <SpeechRecorder onRecorded={handleRecorded} />}
      {result && plotUrl && <SpeechResult result={result} plotUrl={plotUrl} />}
    </div>
  );
}