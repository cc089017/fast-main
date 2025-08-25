from typing import Dict, Tuple
from app.core.config import Modality
from app.services.features.face_features import extract_features_from_image_bytes as face_from_image
from app.services.inference.tabnet_runner import predict_proba_and_label

def run_pipeline(modality: Modality, payload: bytes) -> Tuple[Dict[str, float], float, int]:
    if modality == "face":
        feats, _ = face_from_image(payload)
        if not feats: raise ValueError("얼굴 인식 실패")
    elif modality == "arm":
        raise NotImplementedError("arm pipeline 준비 중")
    elif modality == "speech":
        raise NotImplementedError("speech pipeline 준비 중")
    else:
        raise ValueError("Unknown modality")
    proba, label = predict_proba_and_label(modality, feats)
    return feats, proba, label
