# 모델 아티팩트 로드 및 가드(무결성 검사) - 김민규 작성

import joblib, json, pickle, os, hashlib
import sys
from app.services.speech.quantile_clipper import QuantileClipper
sys.modules['__main__'].QuantileClipper = QuantileClipper

def short_hash(path):
    with open(path, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()[:8]

def load_artifacts(model_dir: str):
    pipe = joblib.load(f"{model_dir}/rf_pipe.joblib")
    xcols = json.load(open(f"{model_dir}/xcols.json"))
    theta = json.load(open(f"{model_dir}/thresholds.json"))["fusion_threshold"]
    with open(f"{model_dir}/dtw_refs.pkl","rb") as f: refs = pickle.load(f)
    meta = json.load(open(f"{model_dir}/meta.json"))
    # 가드: 참조 슬라이스 개수, xcols, meta 등
    assert len(refs)>0 and len(refs[0])==5, "Reference slice count mismatch"
    print(f"[Speech] Artifacts loaded from {model_dir}")
    for fname in ["rf_pipe.joblib", "xcols.json", "thresholds.json", "dtw_refs.pkl", "meta.json"]:
        print(f"  {fname}: {short_hash(os.path.join(model_dir, fname))}")
    return pipe, xcols, theta, refs, meta

# def download_model_if_needed():
#     model_path = "back-end/assets/models/speech_model.joblib"
#     if not os.path.exists(model_path):
#         url = "https://drive.google.com/uc?id=구글드라이브_파일_ID"
#         gdown.download(url, model_path, quiet=False)
#     return model_path

# def download_speech_model_files():
#     import gdown
#     import os
#     files = {
#         "rf_pipe.joblib": "1bKQjPCecmGzbg1Baa3xjEEa-0MpQlpni",
#         "xcols.json": "1UjShIKPd15vriBYApTErLJNdvPopF1NG",
#         "dtw_refs.pkl": "1b2g9EXCKve5zg-b55xzdbBB1DqeykFVo",
#         "meta.json": "1T8g6cgBl-Qlyf894lwuCVmYMHREgOPzy",
#         "thresholds.json": "150GeP1Olbq64l9Gntzs3sxn9wkL-_3nK"
#     }
#     save_dir = "back-end/assets/models/S5_voiced~10s_sr16000_cal_20250910"
#     os.makedirs(save_dir, exist_ok=True)
#     for fname, fid in files.items():
#         url = f"https://drive.google.com/uc?id={fid}"
#         output = os.path.join(save_dir, fname)
#         if not os.path.exists(output):
#             gdown.download(url, output, quiet=False)

# # 모델 로딩 시
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # 4번 올라감
model_path = os.path.join(BASE_DIR, "assets", "models", "S5_voiced~10s_sr16000_cal_20250910", "rf_pipe.joblib")
model = joblib.load(model_path)