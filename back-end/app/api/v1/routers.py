# app/api/v1/routers.py
from fastapi import APIRouter
from app.api.v1.endpoints import auth, measure

api_router = APIRouter()
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(measure.router)  # ✅ prefix 생략
