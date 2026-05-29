"""
Gemini 2.5 Flash 기반 사주 상담 모듈.

- 페르소나별 시스템 프롬프트로 채팅 세션 생성
- 대화 히스토리는 SDK의 Chat 객체가 내부 관리
- 세션 당 최대 MAX_CHAT_TURNS 턴으로 토큰 소비 제한
"""

from __future__ import annotations

import traceback

from google import genai
from google.genai import types

from ai.personas import PERSONA_PROMPTS
from config import MAX_CHAT_TURNS, MODEL_NAME, OHANG_ORDER
from saju.calculator import get_pillar_string

# ──────────────────────────────────────────────
# 싱글턴 클라이언트
# genai.Client 내부 httpx.Client가 GC로 닫히는 것을 방지하기 위해
# 모듈 레벨에서 한 번만 생성하고 재사용한다.
# ──────────────────────────────────────────────
_client: genai.Client | None = None


def _resolve_api_key() -> str:
    """
    런타임에 API 키를 읽는다. 로컬 .env → Streamlit st.secrets 순서로 폴백.

    config.py의 GEMINI_API_KEY와 별도로 런타임에 다시 읽는 이유:
    Streamlit Cloud에서는 앱 초기화 순서 때문에 config.py import 시점에
    st.secrets가 아직 준비되지 않아 빈 문자열이 될 수 있음.
    _get_client() 호출 시점(버튼 클릭 등)에는 st.secrets가 반드시 로드되어 있음.
    """
    import os
    key = os.getenv("GEMINI_API_KEY", "")
    if key:
        return key
    try:
        import streamlit as st
        return st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        return ""


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = _resolve_api_key()
        if not api_key:
            raise ValueError("GEMINI_API_KEY가 설정되지 않았습니다. .env 파일 또는 Streamlit Secrets를 확인하세요.")
        _client = genai.Client(api_key=api_key)
    return _client


def _get_ilgan_info(pillars: dict) -> str:
    """
    일주의 천간(일간)과 오행 반환. 예: '庚(금)'
    일간은 자아를 나타내므로 모든 AI 프롬프트에 명시해 해설 품질을 높임.
    """
    from saju.calculator import OHANG_MAP
    ilju = pillars.get("일주")
    if not ilju:
        return "미상"
    tg = ilju["천간"]
    ohang = OHANG_MAP.get(tg, "")
    return f"{tg}({ohang})"


def _build_user_message(
    pillars: dict,
    ohang_dict: dict[str, int],
    celeb_match: dict | None,
    worry_theme: str,
) -> str:
    """첫 상담 요청 유저 메시지 조립."""
    pillar_str = get_pillar_string(pillars)
    ohang_str = " / ".join(f"{o} {ohang_dict.get(o, 0)}" for o in OHANG_ORDER)
    ilgan = _get_ilgan_info(pillars)

    lines = [
        f"사주 원국: {pillar_str}",
        f"일간(본인): {ilgan}",
        f"오행 점수(지장간·일간 가중치 포함): {ohang_str}",
    ]

    if celeb_match:
        lines.append(
            f"유사 연예인: {celeb_match['name']} ({celeb_match['group']}), "
            f"유사도 {celeb_match['similarity']}%"
        )

    lines.append(f"상담 테마: {worry_theme}")
    # 일간을 "본인의 핵심 기운"으로 명시해 AI가 개인화된 해설을 생성하도록 유도
    lines.append(
        f"일간 {ilgan}을 본인의 핵심 기운으로 삼아, "
        f"오행 에너지 특성과 {worry_theme}에 대한 조언을 해주세요."
    )

    return "\n".join(lines)


def _classify_error(exc: Exception) -> str:
    """예외 → 사용자 표시용 메시지 변환. 기술적 오류를 친화적 문구로 치환."""
    if isinstance(exc, (UnicodeEncodeError, UnicodeDecodeError)):
        return "⚠️ API 키가 올바르지 않습니다. .env 파일에 실제 Gemini API 키를 입력하세요."
    err = str(exc).lower()
    if "429" in err or "quota" in err or "resource_exhausted" in err:
        return "⚠️ AI 사용량이 초과되었습니다. 잠시 후 다시 시도해주세요."
    if "400" in err or "api_key" in err or "invalid" in err:
        return "⚠️ API 키가 유효하지 않습니다. .env 파일의 GEMINI_API_KEY를 확인하세요."
    if "403" in err or "permission" in err:
        return "⚠️ API 접근 권한이 없습니다. API 키 권한을 확인하세요."
    if "timeout" in err or "timed out" in err:
        return "⚠️ 응답 시간이 초과되었습니다. 다시 시도해주세요."
    if "503" in err or "unavailable" in err:
        return "⚠️ AI 서버가 일시적으로 불안정합니다. 잠시 후 다시 시도해주세요."
    return "⚠️ AI 응답 중 오류가 발생했습니다. 다시 시도해주세요."


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

def get_saju_reading(
    persona_key: str,
    pillars: dict,
    ohang_dict: dict[str, int],
    celeb_match: dict | None,
    worry_theme: str,
) -> tuple[str, object | None]:
    """
    첫 번째 Gemini 사주 상담 요청.

    chat 세션을 생성해 반환하는 이유:
    이후 continue_chat()에서 같은 세션으로 추가 질문을 보낼 수 있어
    대화 맥락이 유지됨 (SDK가 히스토리를 내부 관리).

    Args:
        persona_key  : PERSONA_PROMPTS의 키 ("냉철한_선비" 등)
        pillars      : get_saju_pillars() 반환값
        ohang_dict   : {"목": 2, ...} raw 카운트
        celeb_match  : get_top1_info() 반환값 또는 None
        worry_theme  : "직업/진로" 등 상담 테마

    Returns:
        (응답_텍스트, chat_session)
        오류 시 (오류_메시지_문자열, None)
    """
    try:
        client = _get_client()

        system_prompt = PERSONA_PROMPTS.get(persona_key, PERSONA_PROMPTS["따뜻한_조언가"])

        chat = client.chats.create(
            model=MODEL_NAME,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=8192,   # thinking 토큰 소비 감안하여 충분히 확보
                temperature=0.9,
                # thinking_budget=512: 사주 해설은 창의적 글쓰기에 가까워 과도한 추론이 불필요
                thinking_config=types.ThinkingConfig(thinking_budget=512),
            ),
        )

        user_msg = _build_user_message(pillars, ohang_dict, celeb_match, worry_theme)
        response = chat.send_message(user_msg)
        return response.text, chat

    except Exception as exc:
        traceback.print_exc()
        return _classify_error(exc), None


def continue_chat(
    chat_session: object,
    user_message: str,
    current_turn: int,
) -> str:
    """
    기존 chat_session에 추가 질문 전송.

    Args:
        chat_session  : get_saju_reading()에서 반환된 Chat 객체
        user_message  : 사용자 추가 질문 텍스트
        current_turn  : 현재 누적 턴 수 (0부터 시작)

    Returns:
        AI 응답 텍스트 또는 오류 메시지
    """
    if current_turn >= MAX_CHAT_TURNS:
        return f"오늘의 상담 횟수({MAX_CHAT_TURNS}회)를 모두 사용하셨습니다. 내일 다시 찾아주세요. 🙏"

    try:
        response = chat_session.send_message(user_message)
        return response.text
    except Exception as exc:
        traceback.print_exc()
        return _classify_error(exc)


def _build_daily_message(
    pillars: dict,
    ohang_dict: dict[str, int],
    daily_pillars: dict,
    daily_ohang_dict: dict[str, int],
) -> str:
    """오늘의 운세 요청 메시지 조립."""
    from datetime import date as _date
    today = _date.today().strftime("%Y년 %m월 %d일")
    pillar_str = get_pillar_string(pillars)
    daily_str = get_pillar_string(daily_pillars)
    ohang_str = " / ".join(f"{o} {ohang_dict.get(o, 0)}" for o in OHANG_ORDER)
    daily_ohang_str = " / ".join(f"{o} {daily_ohang_dict.get(o, 0)}" for o in OHANG_ORDER)

    ilgan = _get_ilgan_info(pillars)

    return "\n".join([
        f"사주 원국: {pillar_str}",
        f"일간(본인): {ilgan}",
        f"원국 오행: {ohang_str}",
        f"오늘 날짜: {today}",
        f"오늘의 연·월·일주: {daily_str}",
        f"오늘의 오행: {daily_ohang_str}",
        # 200자 내외로 제한: 운세는 간결할수록 가독성이 좋고 토큰도 절약됨
        f"일간 {ilgan}을 본인의 핵심 기운으로 삼아, 원국과 오늘 기운의 상호작용을 200자 내외로 간결하게 요약해줘.",
    ])


def _build_compat_message(
    vec_a: list[float],
    vec_b: list[float],
    pillars_a: dict,
    pillars_b: dict,
    compat_data: dict,
) -> str:
    """궁합 해설 요청 메시지 조립."""
    str_a = get_pillar_string(pillars_a)
    str_b = get_pillar_string(pillars_b)
    # 비율(%)로 표시해 AI가 비중 차이를 직관적으로 파악할 수 있게 함
    ohang_a = " / ".join(f"{o} {round(v*100, 1)}%" for o, v in zip(OHANG_ORDER, vec_a))
    ohang_b = " / ".join(f"{o} {round(v*100, 1)}%" for o, v in zip(OHANG_ORDER, vec_b))
    ss = ", ".join(compat_data["sangseang"]) or "없음"
    sg = ", ".join(compat_data["sanggeuk"]) or "없음"

    ilgan_a = _get_ilgan_info(pillars_a)
    ilgan_b = _get_ilgan_info(pillars_b)

    return "\n".join([
        f"사주 A(나): {str_a}",
        f"A 일간(본인): {ilgan_a}",
        f"오행 A: {ohang_a}",
        f"사주 B(상대): {str_b}",
        f"B 일간(상대): {ilgan_b}",
        f"오행 B: {ohang_b}",
        f"코사인 유사도: {compat_data['similarity']}%",
        f"상생 관계: {ss}",
        f"상극 관계: {sg}",
        f"종합 궁합 점수: {compat_data['score']}%",
        f"A의 일간 {ilgan_a}와 B의 일간 {ilgan_b}의 관계를 중심으로 궁합을 해설해줘. 강점(시너지)과 주의점을 각각 명시하고 600자 내외로 작성해줘.",
    ])


def get_daily_fortune(
    persona_key: str,
    pillars: dict,
    ohang_dict: dict[str, int],
    daily_pillars: dict,
    daily_ohang_dict: dict[str, int],
) -> str:
    """
    오늘의 운세 일회성 Gemini 요청.
    채팅 세션 없이 단일 요청으로 처리 — 대화 이어가기가 필요 없고 토큰을 아낄 수 있음.
    """
    try:
        client = _get_client()
        system_prompt = PERSONA_PROMPTS.get(persona_key, PERSONA_PROMPTS["따뜻한_조언가"])
        user_msg = _build_daily_message(pillars, ohang_dict, daily_pillars, daily_ohang_dict)

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=user_msg,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=1024,  # 운세는 200자 내외이므로 1024로 충분
                temperature=0.9,
                thinking_config=types.ThinkingConfig(thinking_budget=256),  # 짧은 출력, 최소 thinking
            ),
        )
        return response.text
    except Exception as exc:
        traceback.print_exc()
        return _classify_error(exc)


def get_compatibility_reading(
    persona_key: str,
    vec_a: list[float],
    vec_b: list[float],
    pillars_a: dict,
    pillars_b: dict,
    compat_data: dict,
) -> str:
    """
    궁합 해설 일회성 Gemini 요청.
    채팅 세션 없이 단일 요청으로 처리 — 궁합은 1회 해설로 완결되는 컨텐츠.
    """
    try:
        client = _get_client()
        system_prompt = PERSONA_PROMPTS.get(persona_key, PERSONA_PROMPTS["따뜻한_조언가"])
        user_msg = _build_compat_message(vec_a, vec_b, pillars_a, pillars_b, compat_data)

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=user_msg,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=2048,  # 600자 요청이지만 한자·설명 포함 시 여유 필요
                temperature=0.9,
                thinking_config=types.ThinkingConfig(thinking_budget=512),
            ),
        )
        return response.text
    except Exception as exc:
        traceback.print_exc()
        return _classify_error(exc)


def get_chat_history(chat_session: object) -> list[dict[str, str]]:
    """
    Chat 세션의 대화 히스토리를 [{"role": ..., "content": ...}] 형태로 반환.
    Streamlit st.chat_message() 렌더링에 사용.
    """
    history = []
    for content in chat_session.get_history():
        role = "user" if content.role == "user" else "ai"
        text = "".join(part.text for part in content.parts if hasattr(part, "text"))
        if text:
            history.append({"role": role, "content": text})
    return history
