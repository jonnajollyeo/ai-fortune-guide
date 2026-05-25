"""
만세력 기반 사주 팔자 계산 및 오행 벡터 산출 모듈.

sxtwl 라이브러리 동작 규칙 (탐색 검증 완료):
- getYearGZ()  : 입춘(立春) 기준 연주 자동 계산
- getMonthGZ() : 절기(節氣) 기준 월주 자동 계산
- getDayGZ()   : 60갑자 기반 일주 계산
- getHourGZ(h) : 실제 24시간 값(0~23)을 받아 시주 반환
                 야자시(23:00) 처리: 당일 일주 기준 적용 (YAJASI_AS_SAME_DAY=True)
"""

from __future__ import annotations

from datetime import date, datetime

import sxtwl

from config import OHANG_ORDER

# ──────────────────────────────────────────────
# 상수 테이블
# ──────────────────────────────────────────────

CHEONGAN: list[str] = ["甲", "乙", "丙", "丁", "戊", "己", "庚", "辛", "壬", "癸"]
JIJI: list[str] = ["子", "丑", "寅", "卯", "辰", "巳", "午", "未", "申", "酉", "戌", "亥"]

# 천간·지지 → 오행 매핑
OHANG_MAP: dict[str, str] = {
    # 천간
    "甲": "목", "乙": "목",
    "丙": "화", "丁": "화",
    "戊": "토", "己": "토",
    "庚": "금", "辛": "금",
    "壬": "수", "癸": "수",
    # 지지
    "子": "수", "亥": "수",
    "寅": "목", "卯": "목",
    "巳": "화", "午": "화",
    "申": "금", "酉": "금",
    "辰": "토", "戌": "토", "丑": "토", "未": "토",
}

# 시간 표시용 레이블 (UI 선택지용)
HOUR_LABELS: dict[str, int | None] = {
    "모름 (삼주 모드)": None,
    "자시 (23:00~01:00)": 23,
    "축시 (01:00~03:00)": 1,
    "인시 (03:00~05:00)": 3,
    "묘시 (05:00~07:00)": 5,
    "진시 (07:00~09:00)": 7,
    "사시 (09:00~11:00)": 9,
    "오시 (11:00~13:00)": 11,
    "미시 (13:00~15:00)": 13,
    "신시 (15:00~17:00)": 15,
    "유시 (17:00~19:00)": 17,
    "술시 (19:00~21:00)": 19,
    "해시 (21:00~23:00)": 21,
}


# ──────────────────────────────────────────────
# 내부 헬퍼
# ──────────────────────────────────────────────

def _gz_to_dict(tg_idx: int, dz_idx: int) -> dict[str, str]:
    return {"천간": CHEONGAN[tg_idx], "지지": JIJI[dz_idx]}


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

def validate_birth_input(
    year: int, month: int, day: int, hour: int | None
) -> tuple[bool, str]:
    """
    생년월일시 유효성 검사.

    Returns:
        (True, "") 또는 (False, "오류 메시지")
    """
    try:
        birth = date(year, month, day)
    except ValueError:
        return False, f"{year}년 {month}월 {day}일은 존재하지 않는 날짜입니다."

    if year < 1900:
        return False, "1900년 이후의 생년을 입력해주세요."

    if birth > date.today():
        return False, "미래 날짜는 입력할 수 없습니다."

    if hour is not None and not (0 <= hour <= 23):
        return False, f"시간은 0~23 사이여야 합니다. (입력값: {hour})"

    return True, ""


def get_saju_pillars(
    year: int, month: int, day: int, hour: int | None = None
) -> dict[str, dict[str, str] | None]:
    """
    생년월일시 → 사주 팔자(사주 원국) 반환.

    Args:
        year, month, day : 양력 생년월일
        hour             : 24시간제 생시 (0~23). None이면 삼주(三柱) 모드

    Returns:
        {
            "년주": {"천간": "己", "지지": "巳"},
            "월주": {"천간": "丁", "지지": "丑"},
            "일주": {"천간": "庚", "지지": "辰"},
            "시주": {"천간": "辛", "지지": "巳"},  # hour=None이면 None
        }
    """
    day_obj = sxtwl.fromSolar(year, month, day)

    year_gz = day_obj.getYearGZ()
    month_gz = day_obj.getMonthGZ()
    day_gz = day_obj.getDayGZ()

    pillars: dict[str, dict[str, str] | None] = {
        "년주": _gz_to_dict(year_gz.tg, year_gz.dz),
        "월주": _gz_to_dict(month_gz.tg, month_gz.dz),
        "일주": _gz_to_dict(day_gz.tg, day_gz.dz),
        "시주": None,
    }

    if hour is not None:
        hour_gz = day_obj.getHourGZ(hour)
        pillars["시주"] = _gz_to_dict(hour_gz.tg, hour_gz.dz)

    return pillars


def calc_ohang_vector(
    pillars: dict[str, dict[str, str] | None],
) -> tuple[dict[str, int], list[float]]:
    """
    사주 팔자(또는 삼주) → 오행 카운트 및 정규화 벡터 반환.

    Args:
        pillars: get_saju_pillars() 반환값

    Returns:
        ohang_dict   : {"목": 2, "화": 1, "토": 2, "금": 2, "수": 1}
        ohang_vector : [목, 화, 토, 금, 수] L1 정규화 리스트 (합 = 1.0)
    """
    ohang_dict: dict[str, int] = {o: 0 for o in OHANG_ORDER}

    for pillar in pillars.values():
        if pillar is None:
            continue
        for char in (pillar["천간"], pillar["지지"]):
            ohang = OHANG_MAP.get(char)
            if ohang:
                ohang_dict[ohang] += 1

    total = sum(ohang_dict.values())
    if total == 0:
        # 이론상 발생 불가 — 방어 코드
        ohang_vector = [0.2] * 5
    else:
        ohang_vector = [round(ohang_dict[o] / total, 10) for o in OHANG_ORDER]

    return ohang_dict, ohang_vector


def get_pillar_string(pillars: dict[str, dict[str, str] | None]) -> str:
    """팔자 원국을 "己巳 丁丑 庚辰 辛巳" 형태 문자열로 반환. 시주 없으면 3글자쌍만."""
    parts = []
    for key in ["년주", "월주", "일주", "시주"]:
        p = pillars.get(key)
        if p:
            parts.append(p["천간"] + p["지지"])
    return " ".join(parts)


def get_daily_pillar(today: date | None = None) -> tuple[dict, dict[str, int], list[float]]:
    """오늘의 연·월·일주(삼주) 및 오행 벡터 반환. today=None이면 date.today() 사용."""
    if today is None:
        today = date.today()
    pillars = get_saju_pillars(today.year, today.month, today.day, None)
    ohang_dict, ohang_vector = calc_ohang_vector(pillars)
    return pillars, ohang_dict, ohang_vector


def combine_ohang_dicts(
    dict_a: dict[str, int], dict_b: dict[str, int]
) -> tuple[dict[str, int], list[float]]:
    """두 오행 딕셔너리를 합산하여 새 dict와 정규화 벡터 반환."""
    combined = {o: dict_a.get(o, 0) + dict_b.get(o, 0) for o in OHANG_ORDER}
    total = sum(combined.values())
    vector = [round(combined[o] / total, 10) if total > 0 else 0.2 for o in OHANG_ORDER]
    return combined, vector
