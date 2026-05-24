"""
마크다운 리포트 생성 모듈.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from config import OHANG_ORDER
from saju.calculator import get_pillar_string


def generate_report(
    pillars: dict,
    ohang_dict: dict[str, int],
    ohang_vector: list[float],
    celeb_matches: pd.DataFrame,
    ai_reading: str,
) -> str:
    """
    분석 결과를 마크다운 문자열로 조립.

    Args:
        pillars       : get_saju_pillars() 반환값
        ohang_dict    : {"목": 2, ...} raw 카운트
        ohang_vector  : [목, 화, 토, 금, 수] 정규화 벡터
        celeb_matches : find_top_matches() 반환 DataFrame
        ai_reading    : Gemini 생성 사주 해설 텍스트

    Returns:
        마크다운 문자열
    """
    today = date.today().strftime("%Y-%m-%d")
    pillar_str = get_pillar_string(pillars)
    mode = "팔자(사주)" if pillars.get("시주") else "삼주(생시 미입력)"

    lines: list[str] = [
        f"# 나의 사주 에너지 리포트",
        f"",
        f"> 분석일: {today} | 분석 기준: {mode}",
        f"",
        "---",
        "",
        "## 사주 원국",
        "",
        "| 구분 | 천간 | 지지 |",
        "|------|------|------|",
    ]

    for key in ["년주", "월주", "일주", "시주"]:
        p = pillars.get(key)
        if p:
            lines.append(f"| {key} | {p['천간']} | {p['지지']} |")

    lines += [
        "",
        "---",
        "",
        "## 오행 분포",
        "",
        "| 오행 | 글자 수 | 비율 |",
        "|------|---------|------|",
    ]

    ohang_cn = {"목": "木", "화": "火", "토": "土", "금": "金", "수": "水"}
    for o, v in zip(OHANG_ORDER, ohang_vector):
        count = ohang_dict.get(o, 0)
        lines.append(f"| {o}({ohang_cn[o]}) | {count} | {v*100:.1f}% |")

    lines += ["", "---", "", "## 유사 에너지 연예인 TOP3", ""]

    if celeb_matches.empty:
        lines.append("*유사 연예인을 찾지 못했습니다.*")
    else:
        for rank, (_, row) in enumerate(celeb_matches.iterrows(), start=1):
            lines.append(
                f"{rank}. **{row['name']}** ({row['group']}) "
                f"— 유사도 {row['similarity']}%"
            )

    lines += [
        "",
        "---",
        "",
        "## AI 사주 해설",
        "",
        ai_reading,
        "",
        "---",
        "",
        "*본 리포트는 AI 생성 콘텐츠로, 엔터테인먼트 목적입니다.*",
        f"*생성: AI 운명 가이드 | {today}*",
    ]

    return "\n".join(lines)


def get_report_bytes(report_str: str) -> bytes:
    """마크다운 문자열 → UTF-8 bytes (st.download_button용)."""
    return report_str.encode("utf-8")


def get_report_filename() -> str:
    """다운로드 파일명 반환."""
    return f"saju_report_{date.today().strftime('%Y%m%d')}.md"
