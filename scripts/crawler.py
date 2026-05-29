"""
래퍼 생년월일 크롤러.

기본 동작: rappers_manual.csv 로드 (오프라인 시드 데이터).
추가 수집이 필요할 때 crawl_from_web()을 호출하면 나무위키에서 보완 수집을 시도한다.
robots.txt 상 나무위키는 크롤러를 허용하므로 대상으로 선정했다.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
MANUAL_CSV = RAW_DIR / "rappers_manual.csv"
OUTPUT_CSV = RAW_DIR / "rappers_crawled.csv"

# 나무위키 래퍼 목록 문서 (크롤링 대상)
NAMUWIKI_BASE = "https://namu.wiki"
RAPPER_LIST_URL = f"{NAMUWIKI_BASE}/w/한국 힙합 레이블"

# 생년월일 패턴: "19XX년 X월 X일" 또는 "20XX년 X월 X일"
DATE_PATTERN = re.compile(r"(19|20)\d{2}년\s*\d{1,2}월\s*\d{1,2}일")


def load_manual_data() -> pd.DataFrame:
    """수동 시드 CSV 로드. 항상 사용 가능한 기본 데이터."""
    df = pd.read_csv(MANUAL_CSV)
    df["source"] = "manual"
    print(f"[manual] {len(df)}명 로드 완료")
    return df


def _parse_date_string(date_str: str) -> str | None:
    """'19XX년 X월 X일' → 'YYYY-MM-DD' 변환. 실패 시 None."""
    try:
        cleaned = date_str.replace(" ", "")
        m = re.match(r"(\d{4})년(\d{1,2})월(\d{1,2})일", cleaned)
        if not m:
            return None
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return f"{y:04d}-{mo:02d}-{d:02d}"
    except Exception:
        return None


def _get_rapper_birthdate(name: str, session: requests.Session, ua: UserAgent) -> str | None:
    """
    나무위키에서 특정 래퍼 이름으로 생년월일 파싱 시도.
    실패하면 None 반환.
    """
    url = f"{NAMUWIKI_BASE}/w/{requests.utils.quote(name)}"
    try:
        # fake_useragent로 매 요청마다 UA를 바꿔 크롤러 차단 우회
        resp = session.get(url, headers={"User-Agent": ua.random}, timeout=8)
        if resp.status_code != 200:
            return None
        soup = BeautifulSoup(resp.text, "html.parser")
        text = soup.get_text()
        match = DATE_PATTERN.search(text)
        if match:
            return _parse_date_string(match.group())
    except Exception:
        pass
    return None


def crawl_from_web(targets: list[str]) -> pd.DataFrame:
    """
    targets: 추가 수집 대상 래퍼 이름 리스트.
    나무위키에서 생년월일 파싱 후 DataFrame 반환.
    수집 실패 행은 제외.
    """
    session = requests.Session()
    ua = UserAgent()
    rows: list[dict] = []

    for name in targets:
        print(f"[crawl] {name} 수집 중...", end=" ")
        birthdate = _get_rapper_birthdate(name, session, ua)
        if birthdate:
            rows.append({"name": name, "birthdate": birthdate, "group": "래퍼", "source": "crawl"})
            print(f"→ {birthdate}")
        else:
            print("→ 실패 (skip)")
        time.sleep(1.5)  # 서버 부하 방지: 나무위키 이용 정책 준수

    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
        print(f"[crawl] {len(df)}명 수집 완료 → {OUTPUT_CSV}")
    return df


def get_rapper_df(use_web: bool = False, extra_targets: list[str] | None = None) -> pd.DataFrame:
    """
    래퍼 데이터 반환 메인 함수.

    Args:
        use_web      : True면 extra_targets 대상 나무위키 크롤링 추가 실행
        extra_targets: 웹 수집 대상 이름 리스트 (use_web=True 시 사용)

    Returns:
        컬럼: name, birthdate, group, source
    """
    df = load_manual_data()

    if use_web and extra_targets:
        df_web = crawl_from_web(extra_targets)
        if not df_web.empty:
            df = pd.concat([df, df_web], ignore_index=True)

    # name 기준 중복 제거: 수동 데이터와 웹 수집 데이터 간 이름이 겹칠 수 있음
    df = df.drop_duplicates(subset=["name"]).reset_index(drop=True)
    return df


if __name__ == "__main__":
    # 단독 실행 시: 수동 데이터만 로드해서 확인
    df = get_rapper_df()
    print(df.to_string(index=False))
