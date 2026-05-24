import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Gemini API 키: 로컬은 .env, Streamlit Cloud는 st.secrets ──────
def _get_api_key() -> str:
    # 환경변수 우선 (로컬 .env)
    key = os.getenv("GEMINI_API_KEY", "")
    if key:
        return key
    # Streamlit Cloud secrets.toml 폴백
    try:
        import streamlit as st
        return st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        return ""

GEMINI_API_KEY: str = _get_api_key()
MODEL_NAME: str = "gemini-2.5-flash"  # Pro로 변경 금지

# 파일 경로 — __file__ 기준 절대경로 (Streamlit Cloud 호환)
_BASE_DIR = Path(__file__).parent
CELEB_DB_PATH: Path = _BASE_DIR / "data" / "celeb_saju_db.csv"

# ML 파라미터
TOP_N_MATCH: int = 3
SIMILARITY_THRESHOLD: float = 0.5  # 유사도 이 값 미만이면 "해당 없음"

# 채팅 제한
MAX_CHAT_TURNS: int = 5

# 오행 순서 (전체 프로젝트 고정)
OHANG_ORDER: list[str] = ["목", "화", "토", "금", "수"]

# 야자시 처리 방침:
# 23:00~23:59는 당일 일주(日柱) 기준으로 시주 계산
# (명리학 일부 유파에서는 다음날 일주로 산출하기도 하나, 본 프로젝트는 단순화)
YAJASI_AS_SAME_DAY: bool = True
