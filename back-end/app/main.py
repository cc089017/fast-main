# back-end/app/main.py
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles
from starlette.routing import Mount

from app.api.v1.routers import api_router
from app.core.config import settings
from app.db.base import Base
from app.db.session import engine
from app.api.v1.endpoints.arm_predict import router as arm_predict_router


def _resolve_cors_origins() -> list[str]:
    cors = getattr(settings, "CORS_ORIGINS", None)
    if cors:
        if isinstance(cors, str):
            return [o.strip() for o in cors.split(",") if o.strip()]
        if isinstance(cors, (list, tuple)):
            return [str(o) for o in cors]
    origin = getattr(settings, "FRONTEND_ORIGIN", None) or getattr(settings, "FRONTEND_URL", None)
    return [origin or "http://localhost:5173"]


def create_app() -> FastAPI:
    app = FastAPI(title="FAST 프로젝트 백엔드")

    # 개발용: 테이블 자동 생성 (운영은 Alembic 권장)
    Base.metadata.create_all(bind=engine)

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_resolve_cors_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # API v1 및 /predict/ (arm)
    app.include_router(api_router, prefix="/api/v1")
    app.include_router(arm_predict_router)

    # --- STATIC 디버그 & 마운트 (한 번만) ---
    STATIC_DIR = Path(__file__).resolve().parent / "static"
    UPLOAD_DIR = STATIC_DIR / "uploads"
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    try:
        print("[STATIC DEBUG] STATIC_DIR =", STATIC_DIR)
        print("[STATIC DEBUG] exists   =", STATIC_DIR.exists())
        if STATIC_DIR.exists():
            items = [p.name for p in STATIC_DIR.iterdir()]
            print("[STATIC DEBUG] contents:", items)
    except Exception as e:
        print("[STATIC DEBUG] error while listing:", e)

    already_mounted = any(isinstance(r, Mount) and r.path == "/static" for r in app.routes)
    if not already_mounted:
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    # ---------------------------------------

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app


app = create_app()
