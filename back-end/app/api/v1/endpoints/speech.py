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
from app.crud import user as crud_user
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
        os.path.join(root, "assets", "models", "speech"),
        os.path.join(root, "back-end","app", "assets", "models", "speech"),
        os.path.join("assets", "models", "speech"),
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

        # 3.1) 사전 무음 구간 측정(원시 파형 기준) - 침묵/무발화 판정을 위해 사용
        try:
            top_db = float(os.environ.get("SPEECH_NON_SILENT_TOP_DB", "28"))
        except Exception:
            top_db = 28.0
        try:
            non_silent = librosa.effects.split(y_16k, top_db=top_db)
            non_silent_sec_pre = 0.0
            if non_silent is not None and len(non_silent) > 0:
                non_silent_sec_pre = float(sum([(e - s) for s, e in non_silent])) / float(meta.get("sr", 16000))
        except Exception:
            non_silent_sec_pre = 0.0

        # 4) 유성 이어붙이기 + 10s 크롭(최에너지)
        yv = voiced_concat(y_16k, sr=meta.get("sr", 16000))
        print(f"[DEBUG] Voiced concat done, length: {len(yv)}")
        yv = trim_voiced_to_target(yv, meta.get("target_duration_s", 10.0), sr=meta.get("sr", 16000))
        print(f"[DEBUG] Trimmed, length: {len(yv)}")

        # 4.1) (옵션) RMS 정규화: 장비/레벨 차를 완화하기 위해 파일 단위 RMS를 목표로 스케일링
        try:
            rms_norm_flag = (os.environ.get("SPEECH_RMS_NORM", "false") or "false").lower() in ("1","true","yes","on")
            target_rms = float(os.environ.get("SPEECH_RMS_TARGET", "0.073"))
            shrink_only = (os.environ.get("SPEECH_RMS_SHRINK_ONLY", "true") or "true").lower() in ("1","true","yes","on")
        except Exception:
            rms_norm_flag = False
            target_rms = 0.073
            shrink_only = True
        if rms_norm_flag:
            try:
                # zcr_rms와 동일한 파라미터로 RMS 산출(일관성)
                r = librosa.feature.rms(y=yv, frame_length=1024, hop_length=256, center=False)[0]
                cur_rms = float(np.mean(r)) if r is not None and len(r) else 0.0
                if np.isfinite(cur_rms) and cur_rms > 1e-8 and np.isfinite(target_rms) and target_rms > 0:
                    apply = True
                    if shrink_only and cur_rms <= target_rms:
                        apply = False  # 정상보다 작은 RMS는 올리지 않음(위험도 상승 방지)
                    if apply:
                        gain = float(target_rms / cur_rms)
                        yv = (yv * gain).astype(np.float32)
                        print(f"[DEBUG] RMS normalize: cur={cur_rms:.6f} -> target={target_rms:.6f}, gain={gain:.3f}, shrink_only={shrink_only}")
                    else:
                        print(f"[DEBUG] RMS normalize skipped (shrink_only): cur={cur_rms:.6f} <= target={target_rms:.6f}")
                else:
                    print(f"[DEBUG] RMS normalize skipped: cur_rms={cur_rms:.6f}, target={target_rms:.6f}")
            except Exception as e:
                print(f"[WARNING] RMS normalize failed: {e}")

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

        # 5.1) 스펙트럼 평탄도(tonality 판별 지표)
        try:
            flat = librosa.feature.spectral_flatness(y=yv)[0]
            flat_med = float(np.nanmedian(flat)) if flat is not None and len(flat) else float('nan')
        except Exception:
            flat_med = float('nan')

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

        # 피처 중요도(가능한 경우)와 DTW 델타(4.5 기준)를 수집
        try:
            importances = None
            if hasattr(clf, "feature_importances_"):
                importances = [float(x) for x in getattr(clf, "feature_importances_", [])]
            dtw_ref_base = float(os.environ.get("SPEECH_GUARD_DTW_REF", "4.5"))
            dtw_deltas = [float(features.get(f"dtw_slice{i}_mean", float('nan')) - dtw_ref_base) for i in range(1,6)]
        except Exception:
            importances = None
            dtw_ref_base = 4.5
            dtw_deltas = [float('nan')]*5

        # (옵션) 명확한 이상 징후가 있으면(높은 DTW 또는 큰 RMS) 위험도 잠금(lock-in)으로 민감도 보장
        abn_lock_enabled = (os.environ.get("SPEECH_ABN_LOCK_ENABLE", "true") or "true").lower() in ("1","true","yes","on")
        abn_lock_applied = False
        abn_lock_reason = None
        try:
            abn_dtw_margin = float(os.environ.get("SPEECH_ABN_DTW_MARGIN", "1.0"))
            abn_rms_min = float(os.environ.get("SPEECH_ABN_RMS_MIN", "0.11"))
            abn_min_risk = float(os.environ.get("SPEECH_ABN_MIN_RISK", "0.70"))
            dtw_avg_local_for_lock = float(np.nanmean(dtw_means)) if dtw_means else float('nan')
        except Exception:
            abn_dtw_margin = 1.0
            abn_rms_min = 0.11
            abn_min_risk = 0.70
            dtw_avg_local_for_lock = float(np.nanmean(dtw_means)) if dtw_means else float('nan')

        if abn_lock_enabled and np.isfinite(dtw_avg_local if 'dtw_avg_local' in locals() else dtw_avg_local_for_lock):
            try:
                dtw_avg_chk = dtw_avg_local if 'dtw_avg_local' in locals() else dtw_avg_local_for_lock
                dtw_ref_cur = float(os.environ.get("SPEECH_GUARD_DTW_REF", "4.5"))
                cond_dtw_high = np.isfinite(dtw_avg_chk) and (dtw_avg_chk >= (dtw_ref_cur + abn_dtw_margin))
                cond_rms_high = (r_mean is not None) and np.isfinite(r_mean) and (r_mean >= abn_rms_min)
                if cond_dtw_high or cond_rms_high:
                    new_proba = max(float(pos_proba), float(abn_min_risk))
                    if new_proba > pos_proba:
                        print(f"[DEBUG] ABN_LOCK: dtw_avg={dtw_avg_chk:.3f} (>= {dtw_ref_cur+abn_dtw_margin:.3f}) or rms_mean={float(r_mean):.6f} (>= {abn_rms_min:.6f}) -> {pos_proba:.4f}->{new_proba:.4f}")
                        pos_proba = new_proba
                    abn_lock_applied = True
                    abn_lock_reason = {
                        "dtw_avg": dtw_avg_chk,
                        "dtw_ref": dtw_ref_cur,
                        "abn_dtw_margin": abn_dtw_margin,
                        "rms_mean": float(r_mean) if r_mean is not None else None,
                        "abn_rms_min": abn_rms_min,
                        "min_risk": abn_min_risk,
                    }
            except Exception:
                pass

        # (옵션) 침묵/무발화 VETO: 비유성/무발화가 과도하면 정상으로 떨어지지 않도록 최소 위험도 부여
        silence_veto_enabled = (os.environ.get("SPEECH_SILENCE_VETO", "true") or "true").lower() in ("1","true","yes","on")
        silence_veto_applied = False
        silence_veto_reason = None
        try:
            sr_local = int(meta.get("sr", 16000))
            voiced_sec_local = float(len(yv)) / float(sr_local)
            min_voiced_sec = float(os.environ.get("SPEECH_MIN_VOICED_SEC", "2.0"))
            silence_rms_max = float(os.environ.get("SPEECH_SILENCE_RMS_MAX", "0.020"))
            min_risk_when_silence = float(os.environ.get("SPEECH_MIN_RISK_WHEN_SILENCE", "0.65"))
            min_non_silent_sec_pre = float(os.environ.get("SPEECH_MIN_NON_SILENT_SEC", "2.0"))
        except Exception:
            voiced_sec_local = float(len(yv)) / float(meta.get("sr", 16000))
            min_voiced_sec = 2.0
            silence_rms_max = 0.020
            min_risk_when_silence = 0.65
            min_non_silent_sec_pre = 2.0

        if silence_veto_enabled:
            try:
                cond_short = voiced_sec_local < min_voiced_sec or non_silent_sec_pre < min_non_silent_sec_pre
                cond_silent = (r_mean is not None) and np.isfinite(r_mean) and (r_mean <= silence_rms_max)
                if cond_short or cond_silent:
                    new_proba = max(float(pos_proba), float(min_risk_when_silence))
                    if new_proba > pos_proba:
                        print(f"[DEBUG] SILENCE_VETO: voiced_sec={voiced_sec_local:.3f} (< {min_voiced_sec:.3f}) or rms_mean={float(r_mean):.6f} (<= {silence_rms_max:.6f}) -> {pos_proba:.4f}->{new_proba:.4f}")
                        pos_proba = new_proba
                        silence_veto_applied = True
                        silence_veto_reason = {
                            "voiced_sec": voiced_sec_local,
                            "min_voiced_sec": min_voiced_sec,
                            "non_silent_sec_pre": non_silent_sec_pre,
                            "min_non_silent_sec_pre": min_non_silent_sec_pre,
                            "rms_mean": float(r_mean) if r_mean is not None else None,
                            "silence_rms_max": silence_rms_max,
                            "min_risk": min_risk_when_silence,
                        }
            except Exception:
                pass

        # (옵션) 낮은 DTW 하위 꼬리(veto): 평균 DTW가 정상 기준보다 과도하게 낮으면(비정상적 정렬, 음악/배경음 가능성)
        # 위험도 하한을 강제로 올려 threshold(0.538) 이하로 떨어지지 않게 함
        low_dtw_veto_enabled = (os.environ.get("SPEECH_LOW_DTW_VETO", "true") or "true").lower() in ("1","true","yes","on")
        low_dtw_veto_applied = False
        low_dtw_veto_reason = None
        try:
            dtw_avg_local = float(np.nanmean(dtw_means)) if dtw_means else float('nan')
            dtw_ref_floor = float(os.environ.get("SPEECH_GUARD_DTW_REF", "4.5"))
            low_tail_margin = float(os.environ.get("SPEECH_LOW_DTW_MARGIN", "0.2"))
            min_risk_when_low_dtw = float(os.environ.get("SPEECH_MIN_RISK_WHEN_LOW_DTW", "0.60"))
            rms_min_for_veto = float(os.environ.get("SPEECH_VETO_RMS_MIN", "0.04"))
            flatness_max = float(os.environ.get("SPEECH_VETO_FLATNESS_MAX", "0.20"))
        except Exception:
            dtw_ref_floor = 4.5
            low_tail_margin = 0.2
            min_risk_when_low_dtw = 0.60
            rms_min_for_veto = 0.04
            flatness_max = 0.20

        if low_dtw_veto_enabled and np.isfinite(dtw_avg_local):
            try:
                if (dtw_avg_local <= (dtw_ref_floor - low_tail_margin)) and (r_mean is not None) and np.isfinite(r_mean) and (r_mean >= rms_min_for_veto) and (np.isfinite(flat_med) and flat_med <= flatness_max):
                    new_proba = max(float(pos_proba), float(min_risk_when_low_dtw))
                    if new_proba > pos_proba:
                        print(f"[DEBUG] LOW_DTW_VETO: dtw_avg={dtw_avg_local:.3f} <= {dtw_ref_floor-low_tail_margin:.3f}, rms_mean={r_mean:.6f} >= {rms_min_for_veto:.6f}, {pos_proba:.4f}->{new_proba:.4f}")
                        pos_proba = new_proba
                        low_dtw_veto_applied = True
                        low_dtw_veto_reason = {
                            "dtw_avg": dtw_avg_local,
                            "dtw_ref_floor": dtw_ref_floor,
                            "low_tail_margin": low_tail_margin,
                            "rms_mean": float(r_mean),
                            "rms_min_for_veto": rms_min_for_veto,
                            "flatness_med": flat_med,
                            "flatness_max": flatness_max,
                            "min_risk": min_risk_when_low_dtw,
                        }
            except Exception:
                pass

        # (옵션) 매우 낮은 DTW는 명확 이상으로 간주(사용자 요청): dtw_avg <= hard_th 이면 위험도 최소 보장
        low_dtw_hard_enabled = (os.environ.get("SPEECH_LOW_DTW_HARD", "true") or "true").lower() in ("1","true","yes","on")
        low_dtw_hard_applied = False
        low_dtw_hard_reason = None
        try:
            low_dtw_hard_th = float(os.environ.get("SPEECH_LOW_DTW_HARD_TH", "4.0"))
            low_dtw_hard_min_risk = float(os.environ.get("SPEECH_LOW_DTW_HARD_MIN_RISK", "0.75"))
            low_dtw_hard_min_ns = float(os.environ.get("SPEECH_LOW_DTW_HARD_MIN_NON_SILENT", "1.0"))
        except Exception:
            low_dtw_hard_th = 4.0
            low_dtw_hard_min_risk = 0.75
            low_dtw_hard_min_ns = 1.0

        if low_dtw_hard_enabled and np.isfinite(dtw_avg_local):
            try:
                cond_low_dtw = dtw_avg_local <= low_dtw_hard_th
                cond_enough_ns = (non_silent_sec_pre is not None) and (non_silent_sec_pre >= low_dtw_hard_min_ns)
                cond_not_silent_rms = (r_mean is not None) and np.isfinite(r_mean) and (r_mean >= float(os.environ.get("SPEECH_VETO_RMS_MIN", "0.04")))
                if cond_low_dtw and cond_enough_ns and cond_not_silent_rms:
                    new_proba = max(float(pos_proba), float(low_dtw_hard_min_risk))
                    if new_proba > pos_proba:
                        print(f"[DEBUG] LOW_DTW_HARD: dtw_avg={dtw_avg_local:.3f} <= {low_dtw_hard_th:.3f}, non_silent_sec_pre={non_silent_sec_pre:.3f}>= {low_dtw_hard_min_ns:.3f}, rms_mean={r_mean:.6f} -> {pos_proba:.4f}->{new_proba:.4f}")
                        pos_proba = new_proba
                    low_dtw_hard_applied = True
                    low_dtw_hard_reason = {
                        "dtw_avg": dtw_avg_local,
                        "hard_th": low_dtw_hard_th,
                        "non_silent_sec_pre": non_silent_sec_pre,
                        "min_non_silent_sec_pre": low_dtw_hard_min_ns,
                        "rms_mean": float(r_mean) if r_mean is not None else None,
                        "min_risk": low_dtw_hard_min_risk,
                    }
            except Exception:
                pass

        # (옵션) DTW 가드: DTW가 정상 범주이고 RMS도 과도하지 않으면 위험도를 감쇠
        guard_enabled = (os.environ.get("SPEECH_GUARD_ENABLE", "false") or "false").lower() in ("1","true","yes","on")
        guard_applied = False
        guard_reason = None
        try:
            dtw_avg_local = float(np.nanmean(dtw_means)) if dtw_means else float('nan')
            rms_guard_max = float(os.environ.get("SPEECH_GUARD_RMS_MAX", "0.085"))
            dtw_ref = float(os.environ.get("SPEECH_GUARD_DTW_REF", "4.5"))
            dtw_margin = float(os.environ.get("SPEECH_GUARD_MARGIN", "0.3"))
            guard_factor = float(os.environ.get("SPEECH_GUARD_FACTOR", "0.7"))
        except Exception:
            dtw_avg_local = float(np.nanmean(dtw_means)) if dtw_means else float('nan')
            rms_guard_max = 0.085
            dtw_ref = 4.5
            dtw_margin = 0.3
            guard_factor = 0.7

        if guard_enabled and (not low_dtw_veto_applied) and (not silence_veto_applied) and (not low_dtw_hard_applied) and (not abn_lock_applied) and np.isfinite(dtw_avg_local):
            try:
                if (dtw_avg_local <= (dtw_ref + dtw_margin)) and (r_mean is not None) and np.isfinite(r_mean) and (r_mean <= rms_guard_max):
                    new_proba = max(0.0, min(1.0, pos_proba * guard_factor))
                    print(f"[DEBUG] GUARD applied: dtw_avg={dtw_avg_local:.3f}<= {dtw_ref+dtw_margin:.3f}, rms_mean={r_mean:.6f}<= {rms_guard_max:.6f}, {pos_proba:.4f}->{new_proba:.4f}")
                    pos_proba = new_proba
                    guard_applied = True
                    guard_reason = {
                        "dtw_avg": dtw_avg_local,
                        "rms_mean": float(r_mean),
                        "dtw_ref": dtw_ref,
                        "dtw_margin": dtw_margin,
                        "rms_guard_max": rms_guard_max,
                        "factor": guard_factor,
                    }
            except Exception:
                pass

    # (옵션) 하드 가드: 정상 지표에 부합하면 위험도 상한을 강제로 제한
        hard_guard_enabled = (os.environ.get("SPEECH_HARD_GUARD", "false") or "false").lower() in ("1","true","yes","on")
        hard_guard_applied = False
        hard_guard_reason = None
        try:
            hard_margin = float(os.environ.get("SPEECH_HARD_GUARD_MARGIN", os.environ.get("SPEECH_GUARD_MARGIN", "0.6")))
            max_risk_when_normal = float(os.environ.get("SPEECH_MAX_RISK_WHEN_NORMAL", "0.45"))
        except Exception:
            hard_margin = 0.6
            max_risk_when_normal = 0.45

        if hard_guard_enabled and np.isfinite(dtw_avg_local):
            try:
                if (dtw_avg_local <= (dtw_ref + hard_margin)) and (r_mean is not None) and np.isfinite(r_mean) and (r_mean <= rms_guard_max):
                    new_proba = min(float(pos_proba), float(max_risk_when_normal))
                    if new_proba < pos_proba:
                        print(f"[DEBUG] HARD_GUARD cap: dtw_avg={dtw_avg_local:.3f}<= {dtw_ref+hard_margin:.3f}, rms_mean={r_mean:.6f}<= {rms_guard_max:.6f}, cap={max_risk_when_normal:.3f}, {pos_proba:.4f}->{new_proba:.4f}")
                        pos_proba = new_proba
                        hard_guard_applied = True
                        hard_guard_reason = {
                            "dtw_avg": dtw_avg_local,
                            "rms_mean": float(r_mean),
                            "dtw_ref": dtw_ref,
                            "hard_margin": hard_margin,
                            "rms_guard_max": rms_guard_max,
                            "max_risk_when_normal": max_risk_when_normal,
                        }
            except Exception:
                pass

        # 최종 안전망: low_dtw_hard가 위에서 트리거되지 못했더라도, 실제 최종 dtw 평균이 임계 이하이면 한 번 더 강제
        try:
            dtw_avg_final = float(np.nanmean([features.get(f"dtw_slice{i}_mean", float('nan')) for i in range(1,6)]))
        except Exception:
            dtw_avg_final = float('nan')
        if (os.environ.get("SPEECH_LOW_DTW_HARD", "true") or "true").lower() in ("1","true","yes","on"):
            try:
                hard_th2 = float(os.environ.get("SPEECH_LOW_DTW_HARD_TH", "4.0"))
                min_risk2 = float(os.environ.get("SPEECH_LOW_DTW_HARD_MIN_RISK", "0.80"))
                if np.isfinite(dtw_avg_final) and dtw_avg_final <= hard_th2:
                    if pos_proba < min_risk2:
                        print(f"[DEBUG] LOW_DTW_HARD(FINAL): dtw_avg_final={dtw_avg_final:.3f} <= {hard_th2:.3f} -> {pos_proba:.4f}->{min_risk2:.4f}")
                        pos_proba = min_risk2
                        low_dtw_hard_applied = True
                        if not low_dtw_hard_reason:
                            low_dtw_hard_reason = {"dtw_avg": dtw_avg_final, "hard_th": hard_th2, "min_risk": min_risk2, "final_enforced": True}
            except Exception:
                pass

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
                "rms_normalize": rms_norm_flag,
                "rms_target": target_rms,
                "rms_shrink_only": shrink_only,
                "dtw_avg": float(np.nanmean(dtw_means)) if dtw_means else None,
                "dtw_ref": dtw_ref_base,
                "dtw_deltas": dtw_deltas,
                "feature_importances": importances,
                "abn_lock_enabled": abn_lock_enabled,
                "abn_lock_applied": abn_lock_applied,
                "abn_lock_reason": abn_lock_reason,
                "flatness_median": flat_med,
                "guard_enabled": guard_enabled,
                "guard_applied": guard_applied,
                "guard_reason": guard_reason,
                "low_dtw_veto_enabled": low_dtw_veto_enabled,
                "low_dtw_veto_applied": low_dtw_veto_applied,
                "low_dtw_veto_reason": low_dtw_veto_reason,
                "low_dtw_hard_enabled": low_dtw_hard_enabled,
                "low_dtw_hard_applied": low_dtw_hard_applied,
                "low_dtw_hard_reason": low_dtw_hard_reason,
                "silence_veto_enabled": silence_veto_enabled,
                "silence_veto_applied": silence_veto_applied,
                "silence_veto_reason": silence_veto_reason,
                "hard_guard_enabled": hard_guard_enabled,
                "hard_guard_applied": hard_guard_applied,
                "hard_guard_reason": hard_guard_reason,
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
        # 요청된 사용자 안내 문구 형식으로 간결하게 구성
        # 사용자 이름 조회 (없으면 '사용자')
        try:
            user_name = None
            if user_id:
                u = crud_user.get_user_by_id(db, user_id)
                user_name = getattr(u, "name", None) if u is not None else None
        except Exception:
            user_name = None
        name_txt = user_name or "사용자"

        # 설명 문구
        intro_txt = (
            "언어 장애 평가는 MFCC(발음의 정확도)와 정상 참조 음성을 DTW(정상 음성과의 차이)로 비교하여 정상 패턴과의 유사도를 구하고 "
            "ZCR(발성의 불안정성)과 RMS(에너지 일관성)을 함께 고려해 종합 위험도를 산출합니다."
        )
        line2 = (
            f"{name_txt}님의 위험도는 {pos_proba:.2f}로 모델 기준치 {float(theta):.2f} "
            f"{'미만' if pos_proba < float(theta) else '이상'}으로 위험이 "
            f"{'없' if pos_proba < float(theta) else '있'}다고 판단됩니다."
        )
        personalized_text = intro_txt + "\n\n" + line2
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
