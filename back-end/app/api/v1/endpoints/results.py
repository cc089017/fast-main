from __future__ import annotations

from datetime import datetime
from typing import List, Dict

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.core.security import get_user_id_from_cookie
from app.models.speech import Speech
from app.models.face import Face
from app.models.arm import Arm


router = APIRouter(tags=["results"])


def _to_date_str(dt: datetime) -> str:
    if not dt:
        return ""
    return dt.strftime("%Y.%m.%d")


def _speech_to_label(row: Speech) -> str:
    # Speech.result_text: "Abnormal" | "Normal"
    if not row or not row.result_text:
        return "미실시"
    return "경고" if str(row.result_text).lower() == "abnormal" else "정상"


def _face_to_label(row: Face) -> str:
    if not row or not row.result_text:
        return "미실시"
    txt = str(row.result_text)
    # 저장 형태가 문장일 수 있어 부분 매칭
    if "비정상" in txt or "abnormal" in txt.lower():
        return "경고"
    return "정상"


def _arm_to_label(row: Arm) -> str:
    if not row:
        return "미실시"
    # label 1: 경고, 0: 정상 (기본 가정)
    try:
        return "경고" if int(row.label) == 1 else "정상"
    except Exception:
        # 라벨이 문자열이거나 None인 경우
        if str(getattr(row, "label", "")).strip() in ("1", "abnormal", "경고"):
            return "경고"
        if getattr(row, "label", None) is None:
            return "미실시"
        return "정상"


@router.get("/summary")
def get_results_summary(
    limit: int = 20,
    db: Session = Depends(get_db),
    user_id: str = Depends(get_user_id_from_cookie),
) -> List[Dict]:
    """
    사용자별 최근 날짜 단위로 face/arm/speech 결과를 요약해 반환.
    - 날짜는 각 모달리티 레코드의 created_at을 YYYY.MM.DD로 정규화하여 그룹핑
    - 동일 날짜에 여러 건이 있으면 가장 최신(created_at DESC)만 사용
    """
    # 각각 최근 N개 로드
    speech_rows: List[Speech] = (
        db.query(Speech)
        .filter(Speech.user_id == user_id)
        .order_by(Speech.created_at.desc())
        .limit(limit)
        .all()
    )
    face_rows: List[Face] = (
        db.query(Face)
        .filter(Face.user_id == user_id)
        .order_by(Face.created_at.desc())
        .limit(limit)
        .all()
    )
    arm_rows: List[Arm] = (
        db.query(Arm)
        .filter(Arm.user_id == user_id)
        .order_by(Arm.created_at.desc())
        .limit(limit)
        .all()
    )

    # 날짜별 최신 레코드만 유지
    by_date = {}

    for r in speech_rows:
        d = _to_date_str(r.created_at)
        item = by_date.get(d, {"date": d, "face": "미실시", "arm": "미실시", "speech": "미실시"})
        # speech가 더 최신일 수도 있으니 덮어쓰기 허용
        item["speech"] = _speech_to_label(r)
        by_date[d] = item

    for r in face_rows:
        d = _to_date_str(r.created_at)
        item = by_date.get(d, {"date": d, "face": "미실시", "arm": "미실시", "speech": "미실시"})
        if item.get("_face_ts") is None or r.created_at >= item.get("_face_ts"):
            item["face"] = _face_to_label(r)
            item["_face_ts"] = r.created_at
        by_date[d] = item

    for r in arm_rows:
        d = _to_date_str(r.created_at)
        item = by_date.get(d, {"date": d, "face": "미실시", "arm": "미실시", "speech": "미실시"})
        if item.get("_arm_ts") is None or r.created_at >= item.get("_arm_ts"):
            item["arm"] = _arm_to_label(r)
            item["_arm_ts"] = r.created_at
        by_date[d] = item

    # 정렬: 최신 날짜 우선, 내부 메타키 제거
    items = []
    for d, v in by_date.items():
        v.pop("_face_ts", None)
        v.pop("_arm_ts", None)
        items.append(v)

    def _sort_key(it):
        try:
            return datetime.strptime(it["date"], "%Y.%m.%d")
        except Exception:
            return datetime.min

    items.sort(key=_sort_key, reverse=True)
    # 상위 limit로 제한
    return items[:limit]
