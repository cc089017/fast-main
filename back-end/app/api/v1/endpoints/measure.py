# back-end/app/api/v1/endpoints/measure.py
from __future__ import annotations

from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, Request
from sqlalchemy.orm import Session
from typing import Optional, Dict, Any
import json
import logging

from app.db.session import get_db
from app.core.security import get_user_id_from_cookie
from app.schemas.face import FaceOut
from app.crud.face import create_face
from app.services.face_result import compose_result_text
from app.services.pipeline import run_pipeline

# 여기선 prefix 없음. routers.py에서 "/measure"를 붙임
router = APIRouter()


# --- 예측 ---
@router.post("/face/predict")
@router.post("/face/predict/")  # 슬래시 유/무 모두 허용
async def predict_face_only(file: UploadFile = File(...)):
    """
    multipart/form-data: file
    """
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="이미지 파일이 비어 있습니다.")
    try:
        feats, proba, label = run_pipeline("face", data)
    except NotImplementedError as e:
        raise HTTPException(status_code=501, detail=str(e))
    except Exception as e:
        logging.exception("predict_face_only failed")
        raise HTTPException(status_code=400, detail=f"predict_face_only failed: {e}")

    return {
        "modality": "face",
        "pred_proba": float(proba),
        "pred_label": int(label),
        "features": feats,
    }


# --- 업로드/저장 ---
# --- 업로드/저장 ---
@router.post("/face/upload")
@router.post("/face/upload/")  # 슬래시 유/무 모두 허용
async def upload_face_record(
    request: Request,
    db: Session = Depends(get_db),
    image: UploadFile = File(...),
    pred_label: Optional[int] = Form(None),
    features_json: Optional[str] = Form(None),
    result_text: Optional[str] = Form(None),
) -> FaceOut:
    user_id = get_user_id_from_cookie(request)

    # ✅ DB에서 이름 조회
    from app.models.user import User
    user_obj = db.query(User).filter(User.id == user_id).first()
    display_name = user_obj.name if user_obj else user_id  # 이름 없으면 ID fallback

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="이미지 파일이 비어 있습니다.")
    image_mime = image.content_type or "image/jpeg"
    image_size = len(image_bytes)

    # features_json 파싱
    features: Optional[Dict[str, Any]] = None
    if features_json:
        try:
            features = json.loads(features_json)
        except Exception:
            raise HTTPException(status_code=400, detail="features_json 이 유효한 JSON이 아닙니다.")

    # 결과 판단
    is_abnormal: Optional[bool] = None
    if result_text is not None:
        if result_text not in ("정상", "비정상"):
            raise HTTPException(status_code=400, detail="result_text 는 '정상' 또는 '비정상' 이어야 합니다.")
        is_abnormal = (result_text == "비정상")
    elif pred_label is not None:
        if pred_label not in (0, 1):
            raise HTTPException(status_code=400, detail="pred_label 은 0 또는 1 이어야 합니다.")
        is_abnormal = (pred_label == 1)
    else:
        try:
            feats2, proba2, label2 = run_pipeline("face", image_bytes)
            features = features or feats2
            is_abnormal = (int(label2) == 1)
        except Exception as e:
            logging.exception("upload_face_record: re-predict failed")
            raise HTTPException(status_code=400, detail=f"upload_face_record re-predict failed: {e}")

    # 결과 텍스트
    try:
        result_text_final = compose_result_text(
            user_name_or_id=display_name,  # ✅ 이제 이름이 들어감
            is_abnormal=is_abnormal,
            features=features or {},
        )
    except Exception:
        result_text_final = "비정상" if is_abnormal else "정상"

    # DB 저장
    face = create_face(
        db,
        user_id=user_id,
        image_bytes=image_bytes,
        image_mime=image_mime,
        image_size=image_size,
        result_text=result_text_final,
        landmarks_json=features or {},
    )

    return FaceOut(
        face_id=face.face_id,
        user_id=face.user_id,
        result_text=face.result_text,
        created_at=face.created_at,
    )
