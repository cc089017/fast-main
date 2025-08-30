# back-end/app/api/v1/endpoints/arm_predict.py
from __future__ import annotations
import os, time, datetime as dt
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from app.services.features.arm_features import extract_features_from_two_images
from app.services.inference.arm_xgb_runner import predict_proba_and_label

router = APIRouter()

# 업로드를 /static/uploads 로 저장
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # .../app
STATIC_DIR = os.path.join(BASE_DIR, "static")
UPLOAD_DIR = os.path.join(STATIC_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

def _safe_ext(name: str) -> str:
    if not name or "." not in name:
        return ".jpg"
    ext = os.path.splitext(name)[1].lower()
    return ext if ext in {".jpg", ".jpeg", ".png"} else ".jpg"

@router.post("/predict/")
async def predict_arm(start_file: UploadFile = File(...), end_file: UploadFile = File(...)):
    sb = await start_file.read()
    eb = await end_file.read()
    if not sb or not eb:
        raise HTTPException(status_code=400, detail="start_file, end_file 모두 필요합니다.")

    ts = int(time.time() * 1000)
    sname = f"{ts}_t025{_safe_ext(start_file.filename)}"
    ename = f"{ts}_t105{_safe_ext(end_file.filename)}"
    spath = os.path.join(UPLOAD_DIR, sname)
    epath = os.path.join(UPLOAD_DIR, ename)
    with open(spath, "wb") as f:
        f.write(sb)
    with open(epath, "wb") as f:
        f.write(eb)

    feats = extract_features_from_two_images(sb, eb)
    proba, label = predict_proba_and_label(feats)

    # DB 제외: id는 요청 타임스탬프를 그대로 사용(프론트는 단순 표시만 하므로 무방)
    payload = {
        "id": ts,
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "label": label,
        "confidence": round(proba, 6),
        "image_start_url": f"/static/uploads/{sname}",
        "image_end_url": f"/static/uploads/{ename}",
    }
    return JSONResponse(payload)
