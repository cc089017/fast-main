# 모델 아티팩트 로드 및 가드(무결성 검사) - 김민규 작성

import joblib, json, pickle, os, hashlib

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