from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Literal
from app.services.pipeline import run_pipeline

router = APIRouter(prefix="/measure", tags=["measure"])
ModalityParam = Literal["face", "arm", "speech"]

@router.post("/{modality}/upload")
async def upload(modality: ModalityParam, file: UploadFile = File(...)):
    data = await file.read()
    try:
        feats, proba, label = run_pipeline(modality, data)
    except NotImplementedError as e:
        raise HTTPException(501, str(e))
    except Exception as e:
        raise HTTPException(400, str(e))
    return {"modality": modality, "pred_proba": proba, "pred_label": label, "features": feats}
