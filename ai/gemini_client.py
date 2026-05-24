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
from config import GEMINI_API_KEY, MAX_CHAT_TURNS, MODEL_NAME, OHANG_ORDER
from saju.calculator import get_pillar_string

# ──────────────────────────────────────────────
# 싱글턴 클라이언트
# genai.Client 내부 httpx.Client가 GC로 닫히는 것을 방지하기 위해
# 모듈 레벨에서 한 번만 생성하고 재사용한다.
# ──────────────────────────────────────────────
_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


def _build_user_message(
    pillars: dict,
    ohang_dict: dict[str, int],
    celeb_match: dict | None,
    worry_theme: str,
) -> str:
    """첫 상담 요청 유저 메시지 조립."""
    pillar_str = get_pillar_string(pillars)
    ohang_str = " / ".join(f"{o} {ohang_dict.get(o, 0)}" for o in OHANG_ORDER)

    lines = [
        f"사주 원국: {pillar_str}",
        f"오행 점수: {ohang_str}",
    ]

    if celeb_match:
        lines.append(
            f"유사 연예인: {celeb_match['name']} ({celeb_match['group']}), "
            f"유사도 {celeb_match['similarity']}%"
        )

    lines.append(f"상담 테마: {worry_theme}")
    lines.append(
        "위 사주 데이터를 바탕으로 나의 에너지 특성과 "
        f"{worry_theme}에 대한 조언을 해주세요."
    )

    return "\n".join(lines)


def _classify_error(exc: Exception) -> str:
    """예외 → 사용자 표시용 메시지 변환."""
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
                thinking_config=types.ThinkingConfig(thinking_budget=512),  # thinking 최소화
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
