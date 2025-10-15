# Speech API 엔드포인트 - 김민규 작성
# 오디오 파일을 받아서 예측 결과(JSON) 또는 시각화(PNG) 반환

import os
import logging
import uuid
import numpy as np
import subprocess
import shutil
from fastapi import APIRouter, File, UploadFile, HTTPException, Form, Depends
from fastapi.responses import JSONResponse
import librosa
import soundfile as sf
from pydub import AudioSegment
import tempfile
from sqlalchemy.orm import Session
from typing import Optional

from app.services.speech.loader import load_artifacts
from app.services.speech.preprocess import (
    slice_edges, mfcc_cmvn, zcr_rms, dtw_slice_means, dtw_mean_slope,
    voiced_concat, trim_voiced_to_target, load_audio_16k
)
from app.services.speech.viz import build_explain_png
from app.db.session import get_db
from app.crud import speech as crud_speech
from app.core.security import get_user_id_from_cookie

router = APIRouter()

# 모델 디렉터리 해상: 루트(../assets/...)와 back-end/assets 모두 지원
def _resolve_model_dir():
    here = os.path.abspath(os.path.dirname(__file__))
    # 프로젝트 루트 추정 (endpoints -> v1 -> api -> app -> back-end -> ROOT)
    root = os.path.abspath(os.path.join(here, "..", "..", "..", "..", ".."))
    candidates = [
        os.path.join(root, "assets", "models", "S5_voiced~10s_sr16000_cal_20250918_serve4x"),
        os.path.join(root, "back-end", "assets", "models", "S5_voiced~10s_sr16000_cal_20250918_serve4x"),
        os.path.join("assets", "models", "S5_voiced~10s_sr16000_cal_20250918_serve4x"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    # 마지막 수단: 기본 상대경로 반환(존재하지 않을 수 있음)
    return candidates[0]

MODEL_DIR = _resolve_model_dir()

_model_cache = None

def get_model():
    """모델을 캐시와 함께 로드"""
    global _model_cache
    if _model_cache is None:
        _model_cache = load_artifacts(MODEL_DIR)
    return _model_cache

def check_ffmpeg():
    """FFmpeg 설치 확인"""
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def convert_webm_to_wav_ffmpeg(input_path, output_path):
    """FFmpeg를 사용한 WebM → WAV 변환"""
    try:
        cmd = [
            "ffmpeg", 
            "-i", input_path,
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            "-y",
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"[DEBUG] FFmpeg conversion successful: {input_path} → {output_path}")
            return True
        else:
            print(f"[ERROR] FFmpeg failed: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print("[ERROR] FFmpeg not found")
        return False
    except Exception as e:
        print(f"[ERROR] FFmpeg conversion error: {e}")
        return False

def convert_webm_to_wav_pydub(input_path, output_path):
    """pydub를 사용한 WebM → WAV 변환"""
    try:
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(output_path, format="wav")
        print(f"[DEBUG] Pydub conversion successful: {input_path} → {output_path}")
        return True
    except Exception as e:
        print(f"[ERROR] Pydub conversion failed: {e}")
        return False

def convert_webm_to_wav(input_path, output_path):
    """WebM을 WAV로 변환 (FFmpeg 우선, 실패시 pydub)"""
    # FFmpeg 시도
    if check_ffmpeg():
        if convert_webm_to_wav_ffmpeg(input_path, output_path):
            return True
        print("[WARNING] FFmpeg 변환 실패, pydub 시도...")
    
    # pydub 시도
    return convert_webm_to_wav_pydub(input_path, output_path)

@router.get("/system-check")
async def check_system():
    """시스템 환경 확인"""
    return {
        "ffmpeg_installed": check_ffmpeg(),
        "pydub_available": True,
        "model_dir": MODEL_DIR,
        "model_dir_exists": os.path.exists(MODEL_DIR),
        "librosa_available": True,
        "system": "windows",
        "model_files": {
            "rf_pipe": os.path.exists(os.path.join(MODEL_DIR, "rf_pipe.joblib")),
            "thresholds": os.path.exists(os.path.join(MODEL_DIR, "thresholds.json")),
            "meta": os.path.exists(os.path.join(MODEL_DIR, "meta.json")),
            "dtw_refs": os.path.exists(os.path.join(MODEL_DIR, "dtw_refs.pkl"))
        }
    }

@router.get("/model-info")
async def get_model_info():
    """모델 정보 확인"""
    import json
    
    try:
        thresholds_path = os.path.join(MODEL_DIR, "thresholds.json")
        meta_path = os.path.join(MODEL_DIR, "meta.json")
        dtw_scale_path = os.path.join(MODEL_DIR, "dtw_scale.json")
        
        result = {}
        
        if os.path.exists(thresholds_path):
            with open(thresholds_path, 'r', encoding='utf-8') as f:
                thresholds = json.load(f)
                result["thresholds"] = thresholds
        
        if os.path.exists(meta_path):
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
                result["meta"] = meta
        
        if os.path.exists(dtw_scale_path):
            with open(dtw_scale_path, 'r', encoding='utf-8') as f:
                dtw_scale = json.load(f)
                result["dtw_scale"] = dtw_scale
        
        return result
        
    except Exception as e:
        return {"error": str(e)}

@router.post("/predict")
async def predict_speech(
    file: UploadFile = File(...),
    debug: bool = Form(False),
    db: Session = Depends(get_db),
    user_id: str = Depends(get_user_id_from_cookie),
):
    try:
        # 1) 파일 바이트 수신
        b = await file.read()
        if not b:
            return JSONResponse(status_code=400, content={"error": "업로드된 오디오 파일이 비어 있습니다."})
        print(f"[DEBUG] Received file: {file.filename}, size: {len(b)} bytes, type: {file.content_type}")

        # 2) 모델 로드
        pipe, xcols, theta, refs, meta = get_model()
        print(f"[DEBUG] Model loaded - theta: {theta}")

        # 3) 바이트 → 16k mono (webm/mp3/m4a/ogg/wav 모두 대응)
        y_16k = load_audio_16k(b, filename=file.filename)
        print(f"[DEBUG] Audio loaded(16k mono), length: {len(y_16k)}")

        # 4) 유성 이어붙이기 + 10s 크롭(최에너지)
        yv = voiced_concat(y_16k, sr=meta.get("sr", 16000))
        print(f"[DEBUG] Voiced concat done, length: {len(yv)}")
        yv = trim_voiced_to_target(yv, meta.get("target_duration_s", 10.0), sr=meta.get("sr", 16000))
        print(f"[DEBUG] Trimmed, length: {len(yv)}")

        if yv is None or len(yv) < 1024:
            return {
                "error": f"Final audio too short for processing. Length: {len(yv) if yv is not None else 0} samples.",
                "decision": "Error",
                "risk": 0.0,
                "threshold": float(theta),
                "debug": True
            }

        # 5) 특징 추출
        edges = slice_edges(len(yv), 5)
        mfcc_slices = [mfcc_cmvn(yv[s:e], sr=meta["sr"], n_mfcc=13) for (s, e) in edges]
        print(f"[DEBUG] MFCC slices extracted: {len(mfcc_slices)}")

        z_mean, z_std, r_mean, r_std = zcr_rms(yv, sr=meta["sr"])

        scale_mode = os.environ.get("SPEECH_DTW_SCALE_MODE", "auto")
        dtw_means = dtw_slice_means(mfcc_slices, refs, scale_mode=scale_mode)
        dtw_slope = dtw_mean_slope(dtw_means)
        print(f"[DEBUG] DTW means ({scale_mode}): {dtw_means}")
        print(f"[DEBUG] DTW slope: {dtw_slope}")

        features = {f"dtw_slice{i+1}_mean": dtw_means[i] for i in range(5)}
        features.update({
            "dtw_mean_slope": dtw_slope,
            "zcr_mean": z_mean, "zcr_std": z_std,
            "rms_mean": r_mean, "rms_std": r_std
        })

        X = np.array([[features.get(k, np.nan) for k in xcols]], dtype=np.float32)
        pos_proba = float(pipe.predict_proba(X)[0, 1])
        pred = int(pos_proba > float(theta))

        result = {
            "pred": pred,
            "risk": pos_proba,
            "threshold": float(theta),
            "decision": "Normal" if pred == 0 else "Abnormal",
            "features": features,
            "debug_info": {
                "model_theta": float(theta),
                "audio_length": len(yv),
                "audio_duration": len(yv) / meta.get("sr", 16000),
                "dtw_means": dtw_means,
                "dtw_slope": dtw_slope,
                "filename": getattr(file, "filename", None),  # 업로드 파일명
            },
        }

        # 6) 시각화
        extras = {
            "slice_edges": [e[0] for e in edges] + [edges[-1][1]] if edges else [],
            "audio_length": len(yv),
            "audio_duration": len(yv) / meta.get("sr", 16000),
            "slice_count": len(edges),
        }
        try:
            graph_b64 = build_explain_png(
                y=yv, feats=None, features=features, extras=extras, refs=refs,
                meta=meta, risk=pos_proba, theta=float(theta),
                decision=("Abnormal" if pred == 1 else "Normal"),
            )
            result["graph"] = f"data:image/png;base64,{graph_b64}" if graph_b64 else None
            print("[DEBUG] Visualization:", "ok" if graph_b64 else "none")
        except Exception as e:
            print(f"[WARNING] Visualization failed: {e}")
            result["graph"] = None

        # DB 저장 (로그인 필수: cookie 기반 user_id 사용)
        try:
            save_data = {
                "user_id": user_id,
                "wav_filename": getattr(file, "filename", None),
                "model_tag": os.path.basename(MODEL_DIR),
                "risk_score": pos_proba,
                "threshold": float(theta),
                "result_text": "Abnormal" if pred == 1 else "Normal",
                "dtw_slice1": dtw_means[0], "dtw_slice2": dtw_means[1],
                "dtw_slice3": dtw_means[2], "dtw_slice4": dtw_means[3], "dtw_slice5": dtw_means[4],
                "dtw_slope": dtw_slope,
                "x_cover": float(np.isfinite(np.array(list(features.values()), dtype=float)).mean()),
                "voiced_sec": len(yv) / meta.get("sr", 16000),
                "features_json": features,     # 선택
                "debug_json": result.get("debug_info"),  # 선택
            }
            crud_speech.create(db, save_data)
            print("[DEBUG] Speech result saved to DB (user:", user_id, ")")
        except Exception as e:
            print(f"[WARNING] Failed to save speech result: {e}")

        return JSONResponse(result)

    except Exception as e:
        print(f"[ERROR] Speech prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Speech analysis failed: {str(e)}")

@router.get("/list")
def list_speech_results(
    limit: int = 50,
    db: Session = Depends(get_db),
    user_id: str = Depends(get_user_id_from_cookie),
):
    rows = crud_speech.list_by_user(db, user_id=user_id, limit=limit)
    return [
        {
            "id": r.id,
            "created_at": r.created_at,
            "result": r.result_text,
            "risk": r.risk_score,
            "threshold": r.threshold,
            "wav_filename": r.wav_filename,
            "model_tag": r.model_tag,
        } for r in rows
    ]
