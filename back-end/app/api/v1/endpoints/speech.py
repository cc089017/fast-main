# Speech API 엔드포인트 - 김민규 작성
# 오디오 파일을 받아서 예측 결과(JSON) 또는 시각화(PNG) 반환

from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, Response
import os
from app.services.speech.loader import load_artifacts
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))
from app.services.speech.preprocess import load_audio_16k, voiced_slices_and_feats
from app.services.speech.model_adapter import ModelAdapter
from app.services.speech.viz import build_explain_png
from app.services.speech.model_adapter import ModelAdapter
from app.services.speech.viz import build_explain_png

router = APIRouter()

# 환경변수에서 모델 경로 읽기
import os
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
MODEL_DIR = os.getenv("SPEECH_MODEL_DIR", os.path.join(BASE_DIR, 'assets', 'models', 'S5_voiced~10s_sr16000_cal_20250910'))
pipe, xcols, theta, refs, meta = load_artifacts(MODEL_DIR)
model = ModelAdapter(pipe)

UNSURE_BAND = float(os.getenv("UNSURE_BAND", 0.03))

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # 오디오 파일을 16k mono로 로드 및 전처리
        y = load_audio_16k(await file.read())
        feats, mfcc_slices = voiced_slices_and_feats(y)
        # DTW, ZCR/RMS 등 특징 벡터 생성
        from app.services.speech.preprocess import build_feature_vector
        X, features, extras = build_feature_vector(feats, mfcc_slices, refs, xcols)
        # 예측
        risk = float(model.predict_proba_pos(X)[0])
        decision = "Unsure" if abs(risk - theta) <= UNSURE_BAND else ("Abnormal" if risk > theta else "Normal")
        result = {
            "pred": int(risk > theta),
            "risk": risk,
            "threshold": theta,
            "decision": decision,
            "features": features,
            "graph": build_explain_png(y, feats, features, extras, refs, meta, risk, theta, decision)
        }
        return JSONResponse(result)
    except Exception as e:
        import traceback
        print("분석 오류:", e)
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"{type(e).__name__}: {e}")

@router.post("/predict_plot")
async def predict_plot(file: UploadFile = File(...)):
    y = load_audio_16k(await file.read())
    feats, mfcc_slices = voiced_slices_and_feats(y)
    from app.services.speech.preprocess import build_feature_vector
    X, features, extras = build_feature_vector(feats, mfcc_slices, refs, xcols)
    risk = float(model.predict_proba_pos(X)[0])
    decision = "Unsure" if abs(risk - theta) <= UNSURE_BAND else ("Abnormal" if risk > theta else "Normal")
    # 시각화 PNG 생성
    png_bytes = build_explain_png(y, feats, features, extras, refs, meta, risk, theta, decision)
    return Response(content=png_bytes, media_type="image/png")
