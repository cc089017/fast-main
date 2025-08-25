# back-end/app/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.routers import api_router
from app.core.config import settings

# User/인증 등 기존 DB 사용을 위한 import (measurement 관련은 없음)
from app.db.base import Base
from app.db.session import engine


def create_app() -> FastAPI:
    app = FastAPI(title="FAST 프로젝트 백엔드")

    # 개발용: 테이블 자동 생성 (운영은 Alembic 권장)
    Base.metadata.create_all(bind=engine)

    # CORS (프론트 도메인 .env에서 관리)
    origins = [settings.FRONTEND_ORIGIN or "http://localhost:5173"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # API v1
    app.include_router(api_router, prefix="/api/v1")

    # 헬스체크 (옵션)
    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app


app = create_app()
