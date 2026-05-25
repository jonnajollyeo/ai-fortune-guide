"""
AI 운명 가이드 — Streamlit 진입점.
비즈니스 로직은 각 모듈에 위임하고, UI 레이아웃과 세션 흐름만 관리한다.
"""

import streamlit as st

# ── 페이지 설정 (반드시 첫 번째 st 호출) ─────────────────────────
st.set_page_config(
    page_title="AI 운명 가이드",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from datetime import date

from ai.gemini_client import (
    continue_chat,
    get_chat_history,
    get_compatibility_reading,
    get_daily_fortune,
    get_saju_reading,
)
from ai.personas import PERSONA_DESCRIPTIONS, PERSONA_LABELS
from config import CELEB_DB_PATH, MAX_CHAT_TURNS, OHANG_ORDER
from ml.matcher import find_top_matches, format_similarity_label, get_compatibility, get_top1_info, load_celeb_db
from report.generator import generate_report, get_report_bytes, get_report_filename
from saju.calculator import (
    HOUR_LABELS,
    calc_ohang_vector,
    combine_ohang_dicts,
    get_daily_pillar,
    get_pillar_string,
    get_saju_pillars,
    validate_birth_input,
)
from saju.visualizer import draw_ohang_bar, draw_radar_chart, format_ohang_summary


# ── DB 캐시 로드 ──────────────────────────────────────────────────
@st.cache_data
def _load_db():
    return load_celeb_db(CELEB_DB_PATH)


# ── 세션 상태 초기화 ──────────────────────────────────────────────
def _init_session():
    defaults = {
        "pillars": None,
        "ohang_dict": None,
        "ohang_vector": None,
        "matches": None,
        "top1": None,
        "ai_reading": None,
        "chat_session": None,
        "chat_history": [],
        "chat_turn": 0,
        "analyzed": False,
        "persona_key": "따뜻한_조언가",
        # 오늘의 운세
        "daily_fortune": None,
        "daily_analyzed": False,
        # 궁합
        "partner_pillars": None,
        "partner_ohang_dict": None,
        "partner_ohang_vector": None,
        "compat_result": None,
        "compat_reading": None,
        "compat_analyzed": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_session()

# ── 타이틀 ────────────────────────────────────────────────────────
st.title("🔮 AI 운명 가이드")
st.caption("오행 에너지 분석 · 연예인 매칭 · AI 맞춤 해설")

# ════════════════════════════════════════════════════════════════
# 사이드바 — 입력 폼
# ════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("생년월일시 입력")

    with st.form("birth_form"):
        year = st.number_input("태어난 연도", min_value=1900, max_value=2025, value=1995, step=1)
        month = st.selectbox("태어난 월", range(1, 13), index=0)
        day = st.number_input("태어난 일", min_value=1, max_value=31, value=1, step=1)

        hour_label = st.selectbox(
            "태어난 시간",
            list(HOUR_LABELS.keys()),
            index=0,
            help="생시를 모르면 '모름(삼주 모드)'를 선택하세요.",
        )
        hour = HOUR_LABELS[hour_label]

        worry_theme = st.selectbox(
            "상담 테마",
            ["직업/진로", "연애/관계", "재물/금전", "건강/에너지", "종합"],
        )

        st.markdown("**상담사 스타일**")
        persona_labels = list(PERSONA_LABELS.keys())
        persona_label = st.radio(
            "상담사 스타일",
            persona_labels,
            captions=[PERSONA_DESCRIPTIONS[p] for p in persona_labels],
            label_visibility="collapsed",
        )
        persona_key = PERSONA_LABELS[persona_label]

        submitted = st.form_submit_button("✨ 운명 분석 시작", use_container_width=True, type="primary")

    # 분석 실행
    if submitted:
        ok, err_msg = validate_birth_input(int(year), int(month), int(day), hour)
        if not ok:
            st.error(err_msg)
        else:
            # 이전 채팅 및 추가 분석 초기화
            st.session_state.chat_history = []
            st.session_state.chat_turn = 0
            st.session_state.analyzed = False
            st.session_state.daily_fortune = None
            st.session_state.daily_analyzed = False
            st.session_state.compat_result = None
            st.session_state.compat_reading = None
            st.session_state.compat_analyzed = False
            st.session_state.persona_key = persona_key

            with st.spinner("사주를 분석하는 중..."):
                try:
                    df_celeb, matrix = _load_db()

                    pillars = get_saju_pillars(int(year), int(month), int(day), hour)
                    ohang_dict, ohang_vector = calc_ohang_vector(pillars)

                    # 연예인 매칭은 삼주 벡터로 (공정 비교)
                    pillars_3 = get_saju_pillars(int(year), int(month), int(day), None)
                    _, ohang_vector_3 = calc_ohang_vector(pillars_3)
                    matches = find_top_matches(ohang_vector_3, df_celeb, matrix)
                    top1 = get_top1_info(matches)

                    ai_text, chat_session = get_saju_reading(
                        persona_key, pillars, ohang_dict, top1, worry_theme
                    )

                    st.session_state.update({
                        "pillars": pillars,
                        "ohang_dict": ohang_dict,
                        "ohang_vector": ohang_vector,
                        "matches": matches,
                        "top1": top1,
                        "ai_reading": ai_text,
                        "chat_session": chat_session,
                        "analyzed": True,
                    })
                    # 첫 AI 응답을 히스토리에 추가
                    st.session_state.chat_history = [{"role": "ai", "content": ai_text}]

                except Exception as e:
                    st.error(f"분석 중 오류가 발생했습니다: {e}")

# ════════════════════════════════════════════════════════════════
# 메인 화면 — 결과 표시
# ════════════════════════════════════════════════════════════════
if not st.session_state.analyzed:
    # 미분석 상태 안내
    st.info("👈 왼쪽 사이드바에 생년월일을 입력하고 **운명 분석 시작** 버튼을 눌러주세요.")
    st.markdown("""
    ### 이런 걸 알려드려요
    - 🌟 **오행 에너지 분석** — 나의 사주를 5차원 벡터로 시각화
    - 🎤 **연예인 매칭** — 비슷한 에너지를 가진 연예인 TOP 3
    - 🤖 **AI 맞춤 해설** — 선택한 상담사 스타일로 나만의 해설
    - 📄 **리포트 다운로드** — 결과를 마크다운 파일로 저장
    """)
else:
    pillars = st.session_state.pillars
    ohang_dict = st.session_state.ohang_dict
    ohang_vector = st.session_state.ohang_vector
    matches = st.session_state.matches
    top1 = st.session_state.top1
    ai_reading = st.session_state.ai_reading

    # ── 상단: 레이더 차트 + 기본 정보 ────────────────────────────
    col_chart, col_info = st.columns([1, 1], gap="large")

    with col_chart:
        st.subheader("오행 에너지 차트")
        celeb_vec = top1["ohang_vector"] if top1 else None
        celeb_name = top1["name"] if top1 else None
        # 연예인 비교는 삼주 기준 — 표시용 레이블에 명시
        fig_radar = draw_radar_chart(ohang_vector, "나의 에너지", celeb_vec, celeb_name)
        st.plotly_chart(fig_radar, width="stretch")

        fig_bar = draw_ohang_bar(ohang_dict)
        st.plotly_chart(fig_bar, width="stretch")

    with col_info:
        st.subheader("사주 원국")
        pillar_rows = []
        for key in ["년주", "월주", "일주", "시주"]:
            p = pillars.get(key)
            if p:
                pillar_rows.append({"구분": key, "천간": p["천간"], "지지": p["지지"]})
        st.table(pillar_rows)

        st.caption(f"오행 분포: {format_ohang_summary(ohang_dict, ohang_vector)}")

        st.subheader("유사 에너지 연예인 TOP 3")
        if matches is not None and not matches.empty:
            cols = st.columns(len(matches))
            medals = ["🥇", "🥈", "🥉"]
            for col, (rank_idx, row) in zip(cols, matches.iterrows()):
                with col:
                    label = format_similarity_label(row["similarity"])
                    medal = medals[rank_idx] if rank_idx < len(medals) else "·"
                    st.metric(
                        label=f"{medal} {row['name']}",
                        value=f"{row['similarity']}%",
                        delta=label,
                        delta_color="off",
                    )
                    st.caption(row["group"])
        else:
            st.info("유사한 에너지의 연예인을 찾지 못했습니다.")

    st.divider()

    # ── 탭: AI 해설 / 복채 / 오늘의 운세 / 궁합 ──────────────────
    tab_reading, tab_chat, tab_daily, tab_compat = st.tabs([
        "📖 AI 사주 해설", "💬 복채 (추가 질문)", "🌅 오늘의 운세", "💑 궁합 분석"
    ])

    with tab_reading:
        if ai_reading and "⚠️" not in ai_reading:
            st.markdown(ai_reading)
        else:
            st.warning(ai_reading or "AI 해설을 불러오지 못했습니다.")

        # 리포트 다운로드
        if ai_reading and "⚠️" not in ai_reading:
            st.divider()
            report_str = generate_report(
                pillars, ohang_dict, ohang_vector, matches, ai_reading
            )
            st.download_button(
                label="📄 리포트 다운로드 (.md)",
                data=get_report_bytes(report_str),
                file_name=get_report_filename(),
                mime="text/markdown",
            )

    with tab_chat:
        # 채팅 히스토리 표시
        for msg in st.session_state.chat_history:
            role = "assistant" if msg["role"] == "ai" else "user"
            with st.chat_message(role):
                st.markdown(msg["content"])

        # 입력창
        remaining = MAX_CHAT_TURNS - st.session_state.chat_turn
        if st.session_state.chat_session is None:
            st.warning("AI 세션이 없습니다. 분석을 다시 실행해주세요.")
        elif remaining <= 0:
            st.info(f"오늘의 상담 횟수({MAX_CHAT_TURNS}회)를 모두 사용하셨습니다. 🙏")
        else:
            user_input = st.chat_input(
                f"궁금한 점을 더 물어보세요... (남은 횟수: {remaining}회)"
            )
            if user_input:
                # 사용자 메시지 즉시 표시
                st.session_state.chat_history.append({"role": "user", "content": user_input})

                with st.spinner("답변 중..."):
                    reply = continue_chat(
                        st.session_state.chat_session,
                        user_input,
                        st.session_state.chat_turn,
                    )

                st.session_state.chat_history.append({"role": "ai", "content": reply})
                st.session_state.chat_turn += 1
                st.rerun()

    # ── 오늘의 운세 탭 ────────────────────────────────────────────
    with tab_daily:
        st.subheader("🌅 오늘의 운세")

        daily_pillars, daily_ohang_dict, daily_ohang_vector = get_daily_pillar()
        today_str = date.today().strftime("%Y년 %m월 %d일")
        daily_pillar_str = get_pillar_string(daily_pillars)
        st.caption(f"오늘 날짜: {today_str} · 오늘의 연·월·일주: {daily_pillar_str}")

        _, combined_vector = combine_ohang_dicts(ohang_dict, daily_ohang_dict)

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            fig_daily = draw_radar_chart(
                ohang_vector, "나의 원국",
                combined_vector, "원국 + 오늘 기운"
            )
            st.plotly_chart(fig_daily, width="stretch")
        with col_d2:
            st.markdown(f"**오늘의 오행**")
            fig_daily_bar = draw_ohang_bar(daily_ohang_dict)
            st.plotly_chart(fig_daily_bar, width="stretch")

        if not st.session_state.daily_analyzed:
            if st.button("🌅 오늘의 운세 보기", use_container_width=True, type="primary"):
                with st.spinner("오늘의 운세를 분석하는 중..."):
                    fortune = get_daily_fortune(
                        st.session_state.persona_key,
                        pillars,
                        ohang_dict,
                        daily_pillars,
                        daily_ohang_dict,
                    )
                st.session_state.daily_fortune = fortune
                st.session_state.daily_analyzed = True
                st.rerun()

        if st.session_state.daily_analyzed:
            fortune = st.session_state.daily_fortune
            if fortune and "⚠️" not in fortune:
                st.markdown(fortune)
            else:
                st.warning(fortune or "운세를 불러오지 못했습니다.")

    # ── 궁합 분석 탭 ──────────────────────────────────────────────
    with tab_compat:
        st.subheader("💑 사주 궁합 분석")

        with st.form("compat_form"):
            st.markdown("**상대방 생년월일시 입력**")
            p_col1, p_col2, p_col3 = st.columns(3)
            with p_col1:
                p_year = st.number_input("연도", min_value=1900, max_value=2025, value=1995, step=1)
            with p_col2:
                p_month = st.selectbox("월", range(1, 13), index=0)
            with p_col3:
                p_day = st.number_input("일", min_value=1, max_value=31, value=1, step=1)
            p_hour_label = st.selectbox(
                "태어난 시간",
                list(HOUR_LABELS.keys()),
                index=0,
                help="생시를 모르면 '모름(삼주 모드)'를 선택하세요.",
            )
            p_hour = HOUR_LABELS[p_hour_label]
            compat_submitted = st.form_submit_button(
                "💑 궁합 분석 시작", use_container_width=True, type="primary"
            )

        if compat_submitted:
            ok, err = validate_birth_input(int(p_year), int(p_month), int(p_day), p_hour)
            if not ok:
                st.error(err)
            else:
                with st.spinner("궁합을 분석하는 중..."):
                    partner_pillars = get_saju_pillars(int(p_year), int(p_month), int(p_day), p_hour)
                    partner_ohang_dict, partner_ohang_vector = calc_ohang_vector(partner_pillars)

                    compat_result = get_compatibility(ohang_vector, partner_ohang_vector)
                    compat_reading = get_compatibility_reading(
                        st.session_state.persona_key,
                        ohang_vector,
                        partner_ohang_vector,
                        pillars,
                        partner_pillars,
                        compat_result,
                    )

                st.session_state.partner_pillars = partner_pillars
                st.session_state.partner_ohang_dict = partner_ohang_dict
                st.session_state.partner_ohang_vector = partner_ohang_vector
                st.session_state.compat_result = compat_result
                st.session_state.compat_reading = compat_reading
                st.session_state.compat_analyzed = True
                st.rerun()

        if st.session_state.compat_analyzed:
            compat_result = st.session_state.compat_result
            partner_ohang_vector = st.session_state.partner_ohang_vector

            # 점수 요약
            score = compat_result["score"]
            grade = "최상" if score >= 80 else "상" if score >= 65 else "중" if score >= 50 else "하"
            c1, c2, c3 = st.columns(3)
            c1.metric("종합 궁합 점수", f"{score}%")
            c2.metric("오행 유사도", f"{compat_result['similarity']}%")
            c3.metric("궁합 등급", grade)

            # 레이더 차트 비교
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                fig_compat = draw_radar_chart(
                    ohang_vector, "나의 에너지",
                    partner_ohang_vector, "상대 에너지"
                )
                st.plotly_chart(fig_compat, width="stretch")

            with col_c2:
                if compat_result["sangseang"]:
                    st.markdown("**상생 (시너지)**")
                    for s in compat_result["sangseang"]:
                        st.success(s)
                if compat_result["sanggeuk"]:
                    st.markdown("**상극 (주의)**")
                    for s in compat_result["sanggeuk"]:
                        st.warning(s)
                if not compat_result["sangseang"] and not compat_result["sanggeuk"]:
                    st.info("뚜렷한 상생·상극 패턴 없음")

            st.divider()
            compat_reading = st.session_state.compat_reading
            if compat_reading and "⚠️" not in compat_reading:
                st.markdown(compat_reading)
            else:
                st.warning(compat_reading or "궁합 해설을 불러오지 못했습니다.")
