# 설치 가이드 및 Requirements

## 동시 실행을 위한 준비

프로젝트 루트(FAST/)에서 다음 명령을 실행하여 **concurrently**를 설치합니다. 이 모듈은 백엔드와 프론트엔드를 동시에 실행하기 위해 사용

```bash
npm install --save-dev concurrently
```

설치 후

이제 루트에서 `npm start` 명령만으로 백엔드와 프론트엔드를 동시에 기동할 수 있음

---

## 1. 백엔드: Python 의존성 설치

프로젝트 루트(FAST) 기준으로 `requirements.txt`가 위치해 있습니다. 아래 중 한 가지 방법으로 설치하세요.

방법 A) 루트에서 바로 설치

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

방법 B) `back-end` 폴더에서 설치(상대 경로 사용)

```bash
cd back-end
pip install --upgrade pip
pip install -r ../requirements.txt
```

추가 권장 환경 변수(.env, 위험도 완화)

```
SPEECH_VAD_MODE=energy
SPEECH_DTW_SCALE_MODE=auto
SPEECH_THRESHOLD_OVERRIDE=0.65
```

도커 배포 시 DB 설정 전환 예시

```
DB_HOST=mysql
DB_PORT=3306
```

정적 그래프 확인을 위해 Vite 프록시에 `/static`이 FastAPI로 전달되는지 확인하세요.

