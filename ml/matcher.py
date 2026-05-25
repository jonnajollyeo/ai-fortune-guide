"""
코사인 유사도 기반 연예인 오행 매칭 모듈.

학습(training) 없이 벡터 간 수학 연산만으로 유사도 계산.
DB는 앱 기동 시 1회만 로드하며, Streamlit 환경에서는 @st.cache_data로 캐싱.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from config import CELEB_DB_PATH, OHANG_ORDER, SIMILARITY_THRESHOLD, TOP_N_MATCH


def load_celeb_db(path: str = CELEB_DB_PATH) -> tuple[pd.DataFrame, np.ndarray]:
    """
    celeb_saju_db.csv 로드 및 벡터 행렬 구성.

    Returns:
        df     : 원본 DataFrame
        matrix : shape (N, 5) float64 ndarray — 코사인 유사도 계산용
    Raises:
        FileNotFoundError: CSV 파일이 없을 때
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"연예인 DB를 찾을 수 없습니다: {csv_path}\n"
            "scripts/build_celeb_db.py 를 먼저 실행하세요."
        )

    df = pd.read_csv(csv_path)
    df["ohang_vector"] = df["ohang_vector"].apply(json.loads)
    matrix = np.array(df["ohang_vector"].tolist(), dtype=np.float64)  # (N, 5)
    return df, matrix


def find_top_matches(
    user_vector: list[float],
    df: pd.DataFrame,
    matrix: np.ndarray,
    top_n: int = TOP_N_MATCH,
) -> pd.DataFrame:
    """
    사용자 오행 벡터 vs 전체 연예인 벡터 행렬 코사인 유사도 계산.

    Args:
        user_vector : [목, 화, 토, 금, 수] 정규화 벡터 (삼주 기준)
        df          : load_celeb_db() 반환 DataFrame
        matrix      : load_celeb_db() 반환 ndarray
        top_n       : 상위 N명 추출

    Returns:
        상위 N명 DataFrame (컬럼: name, group, birthdate, ohang_vector, similarity)
        유사도 임계값 미달 행은 제거되므로 0행일 수 있음.
    """
    sims: np.ndarray = cosine_similarity([user_vector], matrix)[0]  # shape: (N,)
    top_idx = sims.argsort()[-top_n:][::-1]

    result = df.iloc[top_idx][["name", "group", "birthdate", "ohang_vector"]].copy()
    result["similarity"] = (sims[top_idx] * 100).round(1)
    result = result[result["similarity"] >= SIMILARITY_THRESHOLD * 100]
    return result.reset_index(drop=True)


def get_top1_info(match_df: pd.DataFrame) -> dict | None:
    """
    매칭 결과 중 1위 연예인 정보를 dict로 반환.
    결과가 없으면 None.
    """
    if match_df.empty:
        return None
    row = match_df.iloc[0]
    return {
        "name": row["name"],
        "group": row["group"],
        "birthdate": row["birthdate"],
        "similarity": row["similarity"],
        "ohang_vector": row["ohang_vector"],  # 이미 list 상태
    }


def format_similarity_label(score: float) -> str:
    """유사도 점수 → 한국어 등급 레이블."""
    if score >= 90:
        return "쌍둥이급 에너지"
    if score >= 80:
        return "매우 유사한 에너지"
    if score >= 70:
        return "유사한 에너지"
    if score >= 60:
        return "약간 유사한 에너지"
    return "다른 에너지"


# 상생·상극 관계 테이블 (목→화→토→금→수→목)
_SANGSEANG: dict[str, str] = {"목": "화", "화": "토", "토": "금", "금": "수", "수": "목"}
_SANGGEUK:  dict[str, str] = {"목": "토", "토": "수", "수": "화", "화": "금", "금": "목"}


def get_compatibility(vec_a: list[float], vec_b: list[float]) -> dict:
    """
    두 오행 벡터 간 궁합 점수 및 상생·상극 패턴 분석.

    Returns:
        score      : 종합 궁합 점수 (0~100)
        similarity : 코사인 유사도 (%)
        sangseang  : 상생 관계 설명 리스트
        sanggeuk   : 상극 관계 설명 리스트
    """
    similarity = float(cosine_similarity([vec_a], [vec_b])[0][0]) * 100

    va = dict(zip(OHANG_ORDER, vec_a))
    vb = dict(zip(OHANG_ORDER, vec_b))

    THRESHOLD = 0.25
    sangseang_list: list[str] = []
    sanggeuk_list: list[str] = []
    sangseang_bonus = sanggeuk_penalty = 0.0

    _CN = {"목": "木", "화": "火", "토": "土", "금": "金", "수": "水"}
    # 받침 유무에 따른 조사 (이/가, 을/를)
    _SUBJ = {"목": "이", "화": "가", "토": "가", "금": "이", "수": "가"}
    _OBJ  = {"목": "을", "화": "를", "토": "를", "금": "을", "수": "를"}

    for src, tgt in _SANGSEANG.items():
        if va[src] >= THRESHOLD and vb[tgt] < THRESHOLD:
            sangseang_list.append(
                f"나의 {src}({_CN[src]}){_SUBJ[src]} 상대의 {tgt}({_CN[tgt]}){_OBJ[tgt]} 북돋음"
            )
            sangseang_bonus += va[src]

    for src, tgt in _SANGGEUK.items():
        if va[src] >= THRESHOLD and vb[tgt] >= THRESHOLD:
            sanggeuk_list.append(
                f"나의 {src}({_CN[src]}){_SUBJ[src]} 상대의 {tgt}({_CN[tgt]}){_OBJ[tgt]} 억제"
            )
            sanggeuk_penalty += va[src]

    score = similarity * 0.5 + min(sangseang_bonus * 30, 30) - min(sanggeuk_penalty * 20, 20)
    score = max(0.0, min(100.0, score))

    return {
        "score": round(score, 1),
        "similarity": round(similarity, 1),
        "sangseang": sangseang_list,
        "sanggeuk": sanggeuk_list,
    }
