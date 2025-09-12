# 오디오 전처리 및 특징 추출 모듈 (speech)
# 학습과 동일한 방식으로 16k mono, VAD, MFCC, ZCR/RMS, DTW 등 처리

import io, os, numpy as np, soundfile as sf, librosa
import tempfile
import subprocess
from typing import List, Tuple, Dict

# 기본 파라미터 (meta.json과 일치해야 함)
SR = 16000          # 샘플링 레이트
S = 5               # 슬라이스 개수
N_MFCC = 13         # MFCC 개수

def load_audio_16k(wav_or_bytes) -> np.ndarray:
    """
    bytes 또는 파일 경로를 받아 16kHz mono로 변환
    """
    # webm signature: 1A 45 DF A3
    if isinstance(wav_or_bytes, (bytes, bytearray)):
        # webm signature 체크
        if wav_or_bytes[:4] == b'\x1A\x45\xDF\xA3':
            with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as f_in:
                f_in.write(wav_or_bytes)
                f_in.flush()
                webm_path = f_in.name
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f_out:
                wav_path = f_out.name
            # ffmpeg 변환
            subprocess.run([
                'ffmpeg', '-y', '-i', webm_path, '-ar', '16000', '-ac', '1', wav_path
            ], check=True)
            data, sr = sf.read(wav_path, always_2d=False)
            # 파일 닫힌 뒤 삭제
            os.unlink(webm_path)
            os.unlink(wav_path)
            return data
        else:
            data, sr = sf.read(io.BytesIO(wav_or_bytes), always_2d=False)
            return data
    # path-like
    y, _ = librosa.load(str(wav_or_bytes), sr=SR, mono=True, res_type="kaiser_best")
    return y.astype(np.float32)

def voiced_concat(y, sr=16000, hop_length=256, frame_length=1024):
    """
    VAD: 유성 프레임만 이어 붙이기 (≈10s 목표, 가변 길이 허용)
    """
    f0, vflag, vprob = librosa.pyin(
        y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"),
        sr=sr, frame_length=frame_length, hop_length=hop_length, center=False
    )
    yv_parts = []
    for i, vf in enumerate(vflag):
        if vf:
            s = i * hop_length
            e = s + frame_length
            if s < len(y):
                yv_parts.append(y[s:min(e, len(y))])
    if not yv_parts:
        return np.zeros(0, dtype=np.float32)
    yv = np.concatenate(yv_parts).astype(np.float32)

    # 만약 pyin 실패 시 energy 기반 추출
    intervals = librosa.effects.split(y, top_db=30, frame_length=frame_length, hop_length=hop_length)
    voiced_y = np.concatenate([y[s:e] for s, e in intervals]) if intervals.size else np.array([])
    print("유성음 길이:", len(voiced_y), "전체 길이:", len(y))  # 로그 추가
    return voiced_y

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

def dtw_slice_means(slice_mfcc_list: List[np.ndarray], refs: List[List[np.ndarray]]) -> List[float]:
    """
    각 슬라이스별로 참조(refs)와 DTW 거리 평균 계산
    """
    from fastdtw import fastdtw
    means = []
    for i, mfcc in enumerate(slice_mfcc_list):
        dists = []
        for ref in refs:
            ref_mfcc = ref[i]
            # DTW 거리 계산 (유클리드)
            dist, _ = fastdtw(mfcc.T, ref_mfcc.T, dist=2)
            dists.append(dist)
        means.append(float(np.mean(dists)))
    return means

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

def build_feature_vector(feats: Dict, mfcc_slices: List[np.ndarray], refs: List[List[np.ndarray]], xcols: List[str]) -> Tuple[np.ndarray, Dict, Dict]:
    """
    최종 특징 벡터(xcols 순서), 상세 dict, 추가 정보 반환
    """
    # DTW slice means
    dtw_means = dtw_slice_means(mfcc_slices, refs)
    slope = dtw_mean_slope(dtw_means)
    # 특징 dict
    features = {
        f"dtw_slice{i+1}_mean": dtw_means[i] for i in range(len(dtw_means))
    }
    features["dtw_mean_slope"] = slope
    features["zcr_mean"] = feats["zcr_mean"]
    features["zcr_std"] = feats["zcr_std"]
    features["rms_mean"] = feats["rms_mean"]
    features["rms_std"] = feats["rms_std"]
    # xcols 순서대로 벡터 생성
    X = np.array([[features.get(k, np.nan) for k in xcols]], dtype=np.float32)
    x_cover = float(np.isfinite(X).mean()) / X.shape[1] if X.shape[1] > 0 else 0.0
    extras = {
        "smean": dtw_means,
        "x_cover": x_cover,
        "S_used": feats["S_used"],
        "voiced_sec": len(feats["yv"]) / SR if len(feats["yv"]) > 0 else 0.0
    }
    return X, features, extras
