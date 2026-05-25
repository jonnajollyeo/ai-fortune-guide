"""
연예인 오행 DB 빌드 스크립트 (1회성 실행).

실행 방법:
    cd ai_fortune_guide
    python scripts/build_celeb_db.py

소스 A (선택): data/raw/kpopidolsv3.csv  ← Kaggle에서 수동 다운로드 필요
    URL: https://www.kaggle.com/datasets/nicolsalayoarias/all-kpop-idols
    다운로드 후 data/raw/kpopidolsv3.csv 로 저장

소스 B (기본): data/raw/rappers_manual.csv  ← 수동 시드 데이터 (항상 포함)

출력: data/celeb_saju_db.csv
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# 프로젝트 루트를 sys.path에 추가 (ai_fortune_guide/ 기준 실행)
sys.path.insert(0, str(Path(__file__).parent.parent))

from saju.calculator import get_saju_pillars, calc_ohang_vector
from scripts.crawler import get_rapper_df

RAW_DIR = Path("data/raw")
OUTPUT_PATH = Path("data/celeb_saju_db.csv")
KAGGLE_CSV = RAW_DIR / "kpopidolsv3.csv"


# ──────────────────────────────────────────────
# 소스 A: Kaggle 아이돌 데이터 전처리
# ──────────────────────────────────────────────

def load_kaggle_idols() -> pd.DataFrame:
    """
    kpopidolsv3.csv 전처리.
    파일이 없으면 빈 DataFrame 반환 (소스 B만으로 진행).
    """
    if not KAGGLE_CSV.exists():
        print(f"[skip] {KAGGLE_CSV} 없음 — 래퍼 데이터만으로 진행합니다.")
        print("       Kaggle 데이터를 추가하려면:")
        print("       1. https://www.kaggle.com/datasets/nicolsalayoarias/all-kpop-idols")
        print("       2. 다운로드 후 data/raw/kpopidolsv3.csv 로 저장")
        print("       3. 이 스크립트를 다시 실행\n")
        return pd.DataFrame()

    print(f"[kaggle] {KAGGLE_CSV} 로드 중...")
    df = pd.read_csv(KAGGLE_CSV, encoding="utf-8-sig")

    # 필요 컬럼 추출 및 이름 통일
    col_map = {
        "Stage Name": "name_en",       # 영문 스테이지명 (폴백용)
        "K Stage Name": "name_kr",     # 한글 스테이지명 (우선 사용)
        "Full Name": "full_name",
        "Date of Birth": "birthdate",
        "Group": "group",
    }
    # 실제 컬럼명이 조금 다를 수 있으므로 대소문자 무관 매핑
    df.columns = df.columns.str.strip()
    available = {c.lower(): c for c in df.columns}
    rename = {}
    for want, target in col_map.items():
        key = want.lower()
        if key in available:
            rename[available[key]] = target
    df = df.rename(columns=rename)

    # 필수 컬럼 없으면 종료
    if "birthdate" not in df.columns:
        print("[kaggle] 'Date of Birth' 컬럼을 찾지 못했습니다. CSV 구조를 확인하세요.")
        return pd.DataFrame()

    # name: 한글 스테이지명 우선 → 영문 스테이지명 → Full Name 순 폴백
    name_kr = df.get("name_kr", pd.Series(dtype=str)).str.strip().replace("", pd.NA)
    name_en = df.get("name_en", pd.Series(dtype=str)).str.strip().replace("", pd.NA)
    full    = df.get("full_name", pd.Series(dtype=str)).str.strip().replace("", pd.NA)
    df["name"] = name_kr.fillna(name_en).fillna(full).fillna("Unknown")

    # group 보완
    if "group" in df.columns:
        df["group"] = df["group"].fillna("솔로")
    else:
        df["group"] = "아이돌"

    # 날짜 파싱 — Kaggle CSV는 DD/MM/YYYY 포맷
    df["birthdate"] = pd.to_datetime(df["birthdate"], dayfirst=True, errors="coerce")
    before = len(df)
    df = df.dropna(subset=["birthdate"])
    dropped = before - len(df)
    if dropped:
        print(f"[kaggle] 생년월일 파싱 실패 {dropped}행 제거")

    df["birthdate"] = df["birthdate"].dt.strftime("%Y-%m-%d")
    df["source"] = "kaggle"

    print(f"[kaggle] {len(df)}명 전처리 완료")
    return df[["name", "birthdate", "group", "source"]]


# ──────────────────────────────────────────────
# 오행 벡터 산출
# ──────────────────────────────────────────────

def compute_ohang_vector(birthdate_str: str) -> list[float] | None:
    """
    'YYYY-MM-DD' 문자열 → 삼주 오행 벡터 [목, 화, 토, 금, 수].
    계산 실패 시 None 반환.
    """
    try:
        dt = datetime.strptime(birthdate_str, "%Y-%m-%d")
        pillars = get_saju_pillars(dt.year, dt.month, dt.day, hour=None)
        _, vec = calc_ohang_vector(pillars)
        return vec
    except Exception:
        return None


# ──────────────────────────────────────────────
# 메인 빌드 파이프라인
# ──────────────────────────────────────────────

def build():
    print("=" * 50)
    print("연예인 오행 DB 빌드 시작")
    print("=" * 50)

    # 1. 소스 수집
    df_kaggle = load_kaggle_idols()
    df_rapper = get_rapper_df(use_web=False)
    df_rapper["source"] = df_rapper.get("source", pd.Series("manual", index=df_rapper.index))

    # 2. 병합
    frames = [f for f in [df_kaggle, df_rapper] if not f.empty]
    if not frames:
        print("[error] 데이터 소스가 없습니다.")
        return

    df_all = pd.concat(frames, ignore_index=True)
    print(f"\n[merge] 병합 전 총 {len(df_all)}행")

    # 3. 컬럼 정제
    df_all["name"] = df_all["name"].str.strip()
    df_all["birthdate"] = df_all["birthdate"].str.strip()

    # 4. 중복 제거 (name + birthdate 기준)
    df_all = df_all.drop_duplicates(subset=["name", "birthdate"]).reset_index(drop=True)
    print(f"[merge] 중복 제거 후 {len(df_all)}행")

    # 5. 오행 벡터 일괄 계산 (삼주 기준)
    print("\n[vector] 오행 벡터 산출 중...")
    vectors: list[str | None] = []
    for birthdate in tqdm(df_all["birthdate"], ncols=70):
        vec = compute_ohang_vector(birthdate)
        vectors.append(json.dumps(vec) if vec is not None else None)

    df_all["ohang_vector"] = vectors

    # 계산 실패 행 제거
    before = len(df_all)
    df_all = df_all.dropna(subset=["ohang_vector"]).reset_index(drop=True)
    failed = before - len(df_all)
    if failed:
        print(f"[vector] 계산 실패 {failed}행 제거")

    # 6. 저장
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"\n{'='*50}")
    print(f"완료: {len(df_all)}명 → {OUTPUT_PATH}")
    print(f"소스별 분포:\n{df_all['source'].value_counts().to_string()}")
    print("=" * 50)

    # 7. 간단한 무결성 검사
    _integrity_check(df_all)


def _integrity_check(df: pd.DataFrame):
    """ohang_vector 합산이 1.0인지 샘플 검사."""
    print("\n[check] 무결성 검사 중...")
    sample = df.sample(min(5, len(df)), random_state=42)
    all_ok = True
    for _, row in sample.iterrows():
        vec = json.loads(row["ohang_vector"])
        total = round(sum(vec), 6)
        ok = abs(total - 1.0) < 1e-5
        if not ok:
            print(f"  [FAIL] {row['name']}: 벡터 합 = {total}")
            all_ok = False
    if all_ok:
        print(f"  [PASS] 샘플 {len(sample)}건 벡터 합 모두 1.0")


if __name__ == "__main__":
    build()
