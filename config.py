import os
from pathlib import Path

from dotenv import load_dotenv

# 로컬 개발: .env 파일에서 환경변수 로드 (Streamlit Cloud에서는 secrets.toml 사용)
load_dotenv()

# ── Gemini API 키: 로컬은 .env, Streamlit Cloud는 st.secrets ──────
def _get_api_key() -> str:
    # os.getenv를 먼저 시도하는 이유: 로컬 .env와 Streamlit Cloud 양쪽을 지원하기 위함
    # Streamlit Cloud에서는 st.secrets가 런타임에야 사용 가능하므로 모듈 import 시점에 호출하면 실패할 수 있음
    key = os.getenv("GEMINI_API_KEY", "")
    if key:
        return key
    # 환경변수가 없을 때만 st.secrets를 시도 (Streamlit Cloud 전용 폴백)
    try:
        import streamlit as st
        return st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        return ""

# 모듈 import 시 1회 실행 — 단, gemini_client.py에서도 런타임에 재확인함
# (Streamlit 앱 기동 순서 문제로 config 로딩 시점에 st.secrets가 아직 비어있을 수 있기 때문)
GEMINI_API_KEY: str = _get_api_key()

# Gemini 2.5 Flash를 사용: Pro 대비 비용이 1/5이면서 thinking 기능 지원
MODEL_NAME: str = "gemini-2.5-flash"  # Pro로 변경 금지

# __file__ 기준 절대경로: Streamlit Cloud 배포 시 작업 디렉토리가 달라져도 깨지지 않음
_BASE_DIR = Path(__file__).parent
CELEB_DB_PATH: Path = _BASE_DIR / "data" / "celeb_saju_db.csv"

# 연예인 매칭 파라미터
TOP_N_MATCH: int = 3  # 상위 N명 반환
SIMILARITY_THRESHOLD: float = 0.5  # 이 값 미만이면 "해당 없음" 처리 (코사인 유사도 기준)

# 추가 질문 허용 횟수: 너무 많으면 토큰 소비가 급격히 늘어남
MAX_CHAT_TURNS: int = 5

# 전체 프로젝트에서 오행 순서를 이 상수로 통일 — 벡터 인덱스와 반드시 일치해야 함
OHANG_ORDER: list[str] = ["목", "화", "토", "금", "수"]

# 야자시(23:00~23:59) 처리: 다음날 일주를 쓰는 유파도 있으나
# 일반 사용자 혼란 최소화를 위해 당일 일주 기준으로 단순화
YAJASI_AS_SAME_DAY: bool = True
