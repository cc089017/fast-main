# Speech API 엔드포인트 - 김민규 작성
# 오디오 파일을 받아서 예측 결과(JSON) 또는 시각화(PNG) 반환

import os
import logging
import uuid
import numpy as np
import subprocess
import shutil
import base64
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
    voiced_concat, trim_voiced_to_target, load_audio_16k, load_dtw_scale
)
from app.services.speech.viz import build_explain_png, build_waveform_png, build_dtw_png
from app.db.session import get_db
from app.crud import speech as crud_speech
from app.core.security import get_user_id_from_cookie

router = APIRouter()

# 파이프라인의 최종 분류기 classes_를 가져오는 헬퍼
def _get_classifier_and_classes(model):
    try:
        # sklearn Pipeline인 경우 마지막 스텝이 분류기일 확률이 큼
        if hasattr(model, "steps") and model.steps:
            clf = model.steps[-1][1]
            classes = getattr(clf, "classes_", None)
            return clf, classes
        # 직접 classes_를 갖는 경우
        classes = getattr(model, "classes_", None)
        return model, classes
    except Exception:
        return model, None

# JSON 직렬화/DB 저장을 위한 안전 변환 (numpy → python 기본형, ndarray → list 등)
def _to_py(obj):
    try:
        import numpy as _np
    except Exception:
        _np = None
    # numpy 스칼라
    if _np is not None and isinstance(obj, _np.generic):
        return obj.item()
    # numpy 배열
    if _np is not None and isinstance(obj, _np.ndarray):
        return obj.tolist()
    # 리스트/튜플
    if isinstance(obj, (list, tuple)):
        return [_to_py(x) for x in obj]
    # 딕셔너리: 키도 문자열화
    if isinstance(obj, dict):
        return { (str(k) if not isinstance(k, (str,int,float,bool,type(None))) else k): _to_py(v) for k,v in obj.items() }
    # 기본형
    return obj

# 모델 디렉터리 해상: 루트(../assets/...)와 back-end/assets 모두 지원
def _resolve_model_dir():
    here = os.path.abspath(os.path.dirname(__file__))
    # 프로젝트 루트 추정 (endpoints -> v1 -> api -> app -> back-end -> ROOT)
    root = os.path.abspath(os.path.join(here, "..", "..", "..", "..", ".."))
    candidates = [
        os.path.join(root, "assets", "models", "S5_voiced~10s_sr16000_cal_20250918_serve4x"),
        os.path.join(root, "back-end","app", "assets", "models", "S5_voiced~10s_sr16000_cal_20250918_serve4x"),
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
        # 임계값 환경변수로 오버라이드(테스트/튜닝 편의)
        theta_env = os.environ.get("SPEECH_THRESHOLD_OVERRIDE")
        if theta_env:
            try:
                theta = float(theta_env)
                print(f"[DEBUG] Model loaded - theta overridden by env: {theta}")
            except Exception:
                print(f"[WARNING] Invalid SPEECH_THRESHOLD_OVERRIDE: {theta_env}")
        else:
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
        # DTW 스케일(k)을 올바른 모델 디렉터리에서 읽어 사용해야 학습-서빙이 일치합니다
        dtw_means = dtw_slice_means(mfcc_slices, refs, model_path=MODEL_DIR, scale_mode=scale_mode)
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
        x_cover = float(np.isfinite(X).mean())

        # 분류기 classes_ 기반으로 라벨→확률 매핑 생성
        clf, classes = _get_classifier_and_classes(pipe)
        proba_vec = pipe.predict_proba(X)[0]
        proba_by_label = {}
        if classes is not None:
            for i, lbl in enumerate(list(classes)):
                try:
                    proba_by_label[int(lbl)] = float(proba_vec[i])
                except Exception:
                    # 비정수 라벨이면 문자열 등 그대로 사용
                    proba_by_label[str(lbl)] = float(proba_vec[i])
        else:
            # classes_를 못 찾으면 관례대로 index 1을 abnormal로 간주
            proba_by_label = {0: float(proba_vec[0]) if len(proba_vec) > 0 else float('nan'),
                              1: float(proba_vec[1]) if len(proba_vec) > 1 else float('nan')}

        # 라벨 1을 'Abnormal'로 간주하여 확률 선택
        pos_proba = float(proba_by_label.get(1, proba_by_label.get("1", proba_vec[1] if len(proba_vec) > 1 else float('nan'))))
        pred_label = pipe.predict(X)[0]
        pred = int(pos_proba > float(theta))

        result = {
            "pred": pred,
            "risk": pos_proba,
            "threshold": float(theta),
            "decision": "Normal" if pred == 0 else "Abnormal",
            "features": features,
            "debug_info": {
                "model_theta": float(theta),
                "theta_overridden": bool(theta_env),
                "audio_length": len(yv),
                "audio_duration": len(yv) / meta.get("sr", 16000),
                "dtw_means": dtw_means,
                "dtw_slope": dtw_slope,
                "filename": getattr(file, "filename", None),  # 업로드 파일명
                "dtw_scale_mode": scale_mode,
                "dtw_scale_k": load_dtw_scale(MODEL_DIR),
                "model_dir": MODEL_DIR,
                "x_cover": x_cover,
                "vad_mode": os.environ.get("SPEECH_VAD_MODE", "pyin"),
                "target_duration_s": meta.get("target_duration_s", 10.0),
                "clf_classes": list(classes) if classes is not None else None,
                "proba_vector": [float(x) for x in proba_vec],
                "proba_by_label": proba_by_label,
                "pred_label_raw": int(pred_label) if isinstance(pred_label, (int, np.integer)) else str(pred_label),
            },
        }

        # 6) 시각화 (종합 이미지 + 개별 파형/DTW 그래프)
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

        # 개별 그래프
        wav_b64 = None
        dtw_b64 = None
        try:
            wav_b64 = build_waveform_png(y=yv, meta=meta, extras=extras)
        except Exception as e:
            print(f"[WARNING] Waveform plot failed: {e}")
        try:
            dtw_b64 = build_dtw_png([features.get(f"dtw_slice{i}_mean", 0.0) for i in range(1,6)], normal_ref=4.5)
        except Exception as e:
            print(f"[WARNING] DTW plot failed: {e}")

        # 6.1) 텍스트 리포트 생성(개인화)
        # 첨부 표(연구)의 성능 수치 및 데이터 규모를 문구에 반영
        # - 8000여개의 데이터를 사용하여 학습
        # - 비정상 3972명 중 theta(0.538)보다 risk가 높은 비율 추정: 사용자의 risk가 theta 이상이면 상위 그룹으로 서술
        # - 정상 4240개의 평균 DTW(가정치: 4.5)에 비해 사용자의 평균 DTW가 얼마나 큰지 차이값 표기
        dtw_avg = float(np.mean([features.get(f"dtw_slice{i}_mean", 0.0) for i in range(1,6)]))
        normal_dtw_avg = 4.5  # 시각화에서도 참조하는 기준선(훈련 요약치, 필요시 meta/dtw_scale에서 치환 가능)
        dtw_gap = float(dtw_avg - normal_dtw_avg)
        risk_txt = "경고" if pos_proba > float(theta) else "정상"
        # 퍼센티지 추정 문구 제거: 모델 기준으로 중립적 서술
        personalized_text = (
            f"검사결과 위험도 {pos_proba:.3f}로 ({risk_txt}) 결과가 나왔습니다. "
            f"모델의 임계값 {float(theta):.3f}{'을 초과' if pos_proba > float(theta) else ' 이하'}하였습니다. "
            f"정상 음성 데이터의 평균 DTW 거리(약 {normal_dtw_avg:.2f})에 비해 현재 평균 DTW가 {dtw_gap:+.2f}만큼 {'크' if dtw_gap>=0 else '작'}습니다. "
            "본 결과는 모형 기준 판정으로 임상적 진단을 대체하지 않습니다. 필요 시 재검 혹은 관리가 필요합니다. "
            "*8000여개의 데이터를 사용하여 학습하였음*"
        )
        result["personalized_text"] = personalized_text
        # debug_info에도 삽입해 detail API에서 끌어다 쓸 수 있게 함
        try:
            if isinstance(result.get("debug_info"), dict):
                result["debug_info"]["personalized_text"] = personalized_text
        except Exception:
            pass

        # DB 저장 (로그인 필수: cookie 기반 user_id 사용)
        try:
            safe_features = _to_py(features)
            safe_debug = _to_py(result.get("debug_info"))
            # 저장 디렉터리 (정적 파일로 저장해 URL 제공) - 절대경로로 고정
            # this_file = .../back-end/app/api/v1/endpoints/speech.py
            # app_root = .../back-end/app
            # __file__ = .../back-end/app/api/v1/endpoints/speech.py
            # ../../.. => .../back-end/app
            app_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
            static_dir = os.path.join(app_root, "static", "speech")
            os.makedirs(static_dir, exist_ok=True)
            # 파일명: uuid로 생성
            wav_name = None
            dtw_name = None
            try:
                if wav_b64:
                    wav_name = f"wave_{uuid.uuid4().hex}.png"
                    with open(os.path.join(static_dir, wav_name), "wb") as f:
                        f.write(base64.b64decode(wav_b64))
                if dtw_b64:
                    dtw_name = f"dtw_{uuid.uuid4().hex}.png"
                    with open(os.path.join(static_dir, dtw_name), "wb") as f:
                        f.write(base64.b64decode(dtw_b64))
            except Exception as e:
                print(f"[WARNING] Failed saving speech graphs: {e}")

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
                "x_cover": x_cover,
                "voiced_sec": len(yv) / meta.get("sr", 16000),
                "features_json": safe_features,     # 선택
                "debug_json": safe_debug,  # 선택
                "waveform_graph_url": (f"/static/speech/{wav_name}" if wav_name else None),
                "dtw_graph_url": (f"/static/speech/{dtw_name}" if dtw_name else None),
                "feature_graph_url": None,
            }
            crud_speech.create(db, save_data)
            print("[DEBUG] Speech result saved to DB (user:", user_id, ")")
        except Exception as e:
            print(f"[WARNING] Failed to save speech result: {e}")

        # 최종 응답도 numpy 타입을 정리해서 500 방지
        safe_result = _to_py(result)
        return JSONResponse(safe_result)

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
