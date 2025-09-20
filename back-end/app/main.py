# back-end/app/main.py
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.services.speech.loader import load_artifacts
from app.api.v1.routers import api_router


# 모델 디렉터리 경로 지정 (실제 경로에 맞게 수정)
MODEL_DIR = "C:/fast-main/back-end/assets/models/S5_voiced~10s_sr16000_cal_20250910"

# artifacts 로드
pipe, xcols, theta, refs, meta = load_artifacts(MODEL_DIR)

app = FastAPI()

# CORS 설정 등 기존 코드 유지
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(api_router, prefix="/api/v1")
