from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.schemas.user import UserCreate, UserLogin, TokenResponse, MessageResponse
from app.crud.user import get_user_by_email, create_user, verify_password,get_user_by_id
from app.core.security import create_access_token

router = APIRouter(prefix="/auth", tags=["auth"])

@router.post("/signup", response_model=MessageResponse, status_code=status.HTTP_201_CREATED)
def signup(user_in: UserCreate, db: Session = Depends(get_db)):
    if get_user_by_id(db, user_in.id):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="이미 존재하는 아이디입니다.")
    if get_user_by_email(db, user_in.email):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="이미 존재하는 이메일입니다.")
    create_user(db, user_in)
    return {"message": "회원가입 성공"}

@router.post("/login", response_model=TokenResponse)
def login(user_in: UserLogin, db: Session = Depends(get_db)):
    db_user = get_user_by_id(db, user_in.id)
    if not db_user or not verify_password(user_in.password, db_user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="이메일 또는 비밀번호가 올바르지 않습니다.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token(subject=db_user.id)
    return {"access_token": token}