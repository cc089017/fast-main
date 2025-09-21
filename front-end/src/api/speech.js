// speech.js
export async function sendSpeechAudio(audioBlob) {
  const formData = new FormData();
  formData.append('file', audioBlob, 'speech.webm');
  const res = await fetch('/api/v1/endpoints/speech/predict', {
    method: 'POST',
    body: formData
  });
  return await res.json();
}

export async function getSpeechPlot(audioBlob) {
  const formData = new FormData();
  formData.append('file', audioBlob, 'speech.webm');
  const res = await fetch('/api/v1/endpoints/speech/predict_plot', {
    method: 'POST',
    body: formData
  });
  return await res.blob();
}

export async function saveSpeechResult(data) {
  const res = await fetch('/api/v1/endpoints/speech/save_result', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data)
  });
  return await res.json();
}

export async function predictSpeech(formData) {
  const res = await fetch('/api/v1/endpoints/speech/predict', {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) throw new Error('API Error');
  return await res.json();
}