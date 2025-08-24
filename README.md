# 설치 가이드 및 Requirements

## 동시 실행을 위한 준비

프로젝트 루트(FAST/)에서 다음 명령을 실행하여 **concurrently**를 설치합니다. 이 모듈은 백엔드와 프론트엔드를 동시에 실행하기 위해 사용

```bash
npm install --save-dev concurrently
```

설치 후

이제 루트에서 `npm start` 명령만으로 백엔드와 프론트엔드를 동시에 기동할 수 있음

---

## 1. 백엔드: Python 의존성 (`back-end/requirements.txt`)
```

### 설치 방법

1. 백엔드 디렉토리로 이동:

   ```bash
   cd back-end
   ```


2. 패키지 설치:

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

