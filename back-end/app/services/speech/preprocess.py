# 오디오 전처리 및 특징 추출 모듈 (speech)
# 학습과 동일한 방식으로 16k mono, VAD, MFCC, ZCR/RMS, DTW 등 처리

import io, os, numpy as np, soundfile as sf, librosa
import tempfile
import subprocess
from typing import List, Tuple, Dict
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import numpy as np

# 기본 파라미터 (meta.json과 일치해야 함)
SR = 16000          # 샘플링 레이트
S = 5               # 슬라이스 개수
N_MFCC = 13         # MFCC 개수

def load_audio_16k(wav_or_bytes, filename=None):
    """
    bytes 또는 파일 경로를 받아 16kHz mono로 변환 (webm도 robust하게 지원)
    """
    def _to_16k_mono(y, sr):
        y = librosa.to_mono(y.T) if (y.ndim == 2 and y.shape[1] > 1) else (y if y.ndim==1 else y.squeeze())
        if sr != SR:
            y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=SR, res_type="kaiser_best")
        return y.astype(np.float32)

    if isinstance(wav_or_bytes, (bytes, bytearray, memoryview, io.BytesIO)):
        b = wav_or_bytes if isinstance(wav_or_bytes, (bytes, bytearray, memoryview)) else wav_or_bytes.getvalue()
        # webm 시그니처 체크
        if b[:4] == b'\x1A\x45\xDF\xA3':
            with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as f_in:
                f_in.write(b); f_in.flush(); webm_path = f_in.name
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f_out:
                wav_path = f_out.name
            subprocess.run(['ffmpeg','-y','-i',webm_path,'-ar','16000','-ac','1',wav_path], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            y, sr = sf.read(wav_path, always_2d=False)
            os.unlink(webm_path); os.unlink(wav_path)
            return _to_16k_mono(np.array(y, dtype=np.float32), sr=SR)
        else:
            y, sr = sf.read(io.BytesIO(b), always_2d=False)
            return _to_16k_mono(np.array(y, dtype=np.float32), sr)
    y, sr = sf.read(str(wav_or_bytes), always_2d=False)
    return _to_16k_mono(np.array(y, dtype=np.float32), sr)

def voiced_concat(y, sr=16000, hop_length=256, frame_length=1024):
    """
    VAD: pyin 기반 유성 프레임만 이어 붙이기 (학습과 동일, 실패 시 energy fallback)
    """
    f0, vflag, vprob = librosa.pyin(
        y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"),
        sr=sr, frame_length=frame_length, hop_length=hop_length, center=False
    )
    yv_parts = []
    if vflag is not None and np.any(vflag):
        for i, vf in enumerate(vflag):
            if vf:
                s = i * hop_length; e = s + frame_length
                if s < len(y): yv_parts.append(y[s:min(e, len(y))])
        if yv_parts:
            return np.concatenate(yv_parts).astype(np.float32)

    # pyin 실패 시 energy 기반 fallback
    intervals = librosa.effects.split(y, top_db=30, frame_length=frame_length, hop_length=hop_length)
    if intervals.size:
        return np.concatenate([y[s:e] for s, e in intervals]).astype(np.float32)
    return np.zeros(0, dtype=np.float32)

def slice_edges(L: int, S_: int = S) -> List[Tuple[int, int]]:
    """
    L 길이를 S 구간으로 균등 분할
    """
    idx = np.linspace(0, L, num=S_+1, dtype=int)
    return [(int(idx[i]), int(idx[i+1])) for i in range(S_)]

def mfcc_cmvn(y_seg: np.ndarray, sr: int = SR, n_mfcc: int = N_MFCC) -> np.ndarray:
    """
    MFCC 추출 + CMVN (슬라이스별)
    """
    if len(y_seg) == 0:
        return np.zeros((n_mfcc, 1), dtype=np.float32)
    M = librosa.feature.mfcc(y=y_seg, sr=sr, n_mfcc=n_mfcc)
    M = (M - M.mean(axis=1, keepdims=True)) / (M.std(axis=1, keepdims=True) + 1e-8)
    return M.astype(np.float32)

def zcr_rms(yv: np.ndarray, sr: int = SR) -> Tuple[float, float, float, float]:
    """
    ZCR/RMS (전체 구간)
    """
    if len(yv) == 0:
        return (np.nan, np.nan, np.nan, np.nan)
    z = librosa.feature.zero_crossing_rate(yv)[0]
    r = librosa.feature.rms(y=yv, frame_length=1024, hop_length=256)[0]
    return float(np.mean(z)), float(np.std(z)), float(np.mean(r)), float(np.std(r))

def voiced_slices_and_feats(y: np.ndarray, sr: int = SR, S_: int = S, n_mfcc: int = N_MFCC):
    """
    한 샘플 → 5 슬라이스 MFCC 리스트, ZCR/RMS, meta
    """
    yv = voiced_concat(y, sr=sr)
    edges = slice_edges(len(yv), S_)
    mfcc_slices = [mfcc_cmvn(yv[s:e], sr=sr, n_mfcc=n_mfcc) for (s, e) in edges]
    z_mean, z_std, r_mean, r_std = zcr_rms(yv, sr=sr)
    return dict(
        yv=yv, edges=edges, S_used=S_,
        zcr_mean=z_mean, zcr_std=z_std, rms_mean=r_mean, rms_std=r_std
    ), mfcc_slices

def dtw_distance_train_like(M1, M2):
    """
    학습과 동일하게 DTW 경로 길이로 나눈 평균 거리 반환
    """
    s1 = M1.T.astype(float)
    s2 = M2.T.astype(float)
    dist, path = fastdtw(s1, s2, dist=euclidean)
    return float(dist) / max(1, len(path))

def dtw_slice_means(slice_mfcc_list: list, refs: list) -> list:
    """
    각 슬라이스별로 참조(refs)와 DTW 거리 평균 계산 (학습과 동일 스케일)
    """
    S = len(slice_mfcc_list)
    dtw_means = []
    for i in range(S):
        dists = []
        for r in refs:
            try:
                d = dtw_distance_train_like(slice_mfcc_list[i], r[i])
            except Exception as e:
                print(f"[WARN] DTW 계산 오류: {e}")
                d = np.nan
            dists.append(d)
        # NaN이 있으면 평균에서 제외
        dists = [x for x in dists if np.isfinite(x)]
        dtw_means.append(float(np.mean(dists)) if dists else np.nan)
    return dtw_means

def dtw_mean_slope(dtw_means: List[float]) -> float:
    """
    5개 DTW 평균의 선형 기울기
    """
    x = np.arange(len(dtw_means))
    y = np.array(dtw_means)
    if len(y) != 5:
        return float('nan')
    coef = np.polyfit(x, y, 1)
    return float(coef[0])

def build_feature_vector(feats, mfcc_slices, refs, xcols):
    S = len(mfcc_slices)
    features = {}
    # 슬라이스별 DTW 평균 계산
    dtw_means = dtw_slice_means(mfcc_slices, refs)
    for i in range(S):
        features[f"dtw_slice{i+1}_mean"] = dtw_means[i]
    slope = dtw_mean_slope(dtw_means)
    features["dtw_mean_slope"] = slope
    features["zcr_mean"] = feats.get("zcr_mean", np.nan)
    features["zcr_std"] = feats.get("zcr_std", np.nan)
    features["rms_mean"] = feats.get("rms_mean", np.nan)
    features["rms_std"] = feats.get("rms_std", np.nan)
    X = np.array([[features.get(k, np.nan) for k in xcols]], dtype=np.float32)
    x_cover = float(np.isfinite(X).mean())  # 분모로 나누지 않음 (1.00이어야 정상)
    extras = {
        "smean": dtw_means,
        "x_cover": x_cover,
        "S_used": feats.get("S_used", S),
        "voiced_sec": len(feats.get("yv", [])) / SR if len(feats.get("yv", [])) > 0 else 0.0
    }
    return X, features, extras

def sanity_check_pipeline(yv, refs, xcols, feats):
    import numpy as np
    try:
        assert len(refs)>0 and len(refs[0])==5, "refs S != 5"
        smeans = [feats.get(f"dtw_slice{i}_mean", np.nan) for i in range(1,6)]
        if np.nanmax(smeans) > 100:
            print("[WARN] DTW scale drift (>100). Check normalization / CMVN / n_mfcc / DTW impl.")
        x_cover = float(np.isfinite(np.array([feats[k] for k in xcols])).mean())
        print(f"[CHK] x_cover={x_cover:.2f} (1.00이어야 정상)")
    except Exception as e:
        print(f"[ERROR] sanity_check_pipeline: {e}")

SR = 16000

def trim_voiced_to_target(yv: np.ndarray, sr: int = SR, min_sec=10, max_sec=15) -> np.ndarray:
    """
    유성음 구간을 최소 10초, 최대 15초로 자릅니다.
    """
    n_min = int(min_sec * sr)
    n_max = int(max_sec * sr)
    if len(yv) < n_min:
        return np.array([], dtype=np.float32)
    if len(yv) > n_max:
        return yv[:n_max]
    return yv
