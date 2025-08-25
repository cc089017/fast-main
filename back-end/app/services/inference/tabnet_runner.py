import os, json, joblib, numpy as np
from functools import lru_cache
from typing import Dict, List, Tuple
from pytorch_tabnet.tab_model import TabNetClassifier
from app.core.config import model_dir, threshold, Modality

@lru_cache(maxsize=8)
def _load_scaler_and_model(modality: Modality):
    mdir = model_dir(modality)
    scaler = joblib.load(os.path.join(mdir, "scaler.save"))
    model = TabNetClassifier()
    model.load_model(os.path.join(mdir, "TabNet_final.zip"))

    fo_path = os.path.join(mdir, "feature_order.json")
    if not os.path.exists(fo_path):
        raise RuntimeError(f"feature_order.json not found in: {mdir}")
    with open(fo_path, "r", encoding="utf-8") as f:
        feature_order: List[str] = json.load(f)

    # 스케일러 입력 차원과 일치 확인(가능한 경우)
    n_scaler = getattr(scaler, "n_features_in_", None) or len(getattr(scaler, "scale_", [])) or len(feature_order)
    if n_scaler != len(feature_order):
        raise RuntimeError(f"feature_order ({len(feature_order)}) != scaler expects ({n_scaler})")

    return scaler, model, feature_order

def predict_proba_and_label(modality: Modality, feats: Dict[str, float]) -> Tuple[float, int]:
    scaler, model, feature_order = _load_scaler_and_model(modality)
    X = np.array([[float(feats.get(k, 0.0)) for k in feature_order]], dtype=float)
    Xs = scaler.transform(X)
    p1 = float(model.predict_proba(Xs)[0, 1])
    return p1, int(p1 >= threshold(modality))
