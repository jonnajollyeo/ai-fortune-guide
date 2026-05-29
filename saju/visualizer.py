"""
Plotly 기반 오행 시각화 모듈.
"""

from __future__ import annotations

import plotly.graph_objects as go

from config import OHANG_ORDER

# 레이더 차트는 마지막 값이 첫 값과 같아야 도형이 닫힘 → 목(木)을 처음과 끝에 배치
_THETA = ["목(木)", "화(火)", "토(土)", "금(金)", "수(水)", "목(木)"]

# 오행별 고유 색상: 전통 오행 색 배정에서 유래 (목=청록, 화=적, 토=황, 금=백/회, 수=흑/청)
OHANG_COLORS = {
    "목": "#4CAF50",  # 초록
    "화": "#F44336",  # 빨강
    "토": "#FFC107",  # 노랑
    "금": "#9E9E9E",  # 회색
    "수": "#2196F3",  # 파랑
}


def draw_radar_chart(
    user_vector: list[float],
    user_label: str = "나의 에너지",
    celeb_vector: list[float] | None = None,
    celeb_label: str | None = None,
) -> go.Figure:
    """
    오행 레이더 차트 생성.

    Args:
        user_vector  : [목, 화, 토, 금, 수] 정규화 벡터
        user_label   : 사용자 트레이스 레이블
        celeb_vector : 연예인 벡터 (None이면 단독 표시)
        celeb_label  : 연예인 트레이스 레이블

    Returns:
        Plotly Figure 객체 — st.plotly_chart()에 직접 전달 가능
    """
    # 두 벡터 중 최댓값 기준으로 축 범위 설정: 어느 쪽도 잘리지 않게 여유 35% 확보
    all_values = user_vector + (celeb_vector or [])
    r_max = max(all_values) * 1.35 if all_values else 0.5

    fig = go.Figure()

    # 사용자 트레이스: 파란 반투명 채움
    user_r = user_vector + [user_vector[0]]  # 도형 닫기 위해 첫 값 반복
    fig.add_trace(go.Scatterpolar(
        r=user_r,
        theta=_THETA,
        fill="toself",
        fillcolor="rgba(65, 105, 225, 0.25)",
        line=dict(color="royalblue", width=2),
        name=user_label,
        hovertemplate="%{theta}: %{r:.3f}<extra></extra>",
    ))

    # 연예인 비교 트레이스: 빨간 점선으로 구분 (있을 때만 추가)
    if celeb_vector is not None:
        celeb_r = celeb_vector + [celeb_vector[0]]
        fig.add_trace(go.Scatterpolar(
            r=celeb_r,
            theta=_THETA,
            fill="toself",
            fillcolor="rgba(220, 20, 60, 0.2)",
            line=dict(color="crimson", width=2, dash="dot"),
            name=celeb_label or "연예인",
            hovertemplate="%{theta}: %{r:.3f}<extra></extra>",
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, r_max],
                tickfont=dict(size=10),
                gridcolor="lightgray",
            ),
            angularaxis=dict(
                tickfont=dict(size=13),
            ),
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,   # 차트 아래 범례 배치 (차트 내부 겹침 방지)
            xanchor="center",
            x=0.5,
        ),
        margin=dict(t=30, b=60, l=40, r=40),
        height=380,
    )
    return fig


def draw_ohang_bar(ohang_dict: dict[str, int]) -> go.Figure:
    """
    오행 raw 카운트 막대 차트 생성.

    Args:
        ohang_dict: {"목": 2, "화": 1, "토": 2, "금": 2, "수": 1}
    """
    # 한글 + 한자를 같이 표시: "목(木)" 형태로 가독성 향상
    labels = [f"{o}({['木','火','土','金','水'][i]})" for i, o in enumerate(OHANG_ORDER)]
    values = [ohang_dict.get(o, 0) for o in OHANG_ORDER]
    colors = [OHANG_COLORS[o] for o in OHANG_ORDER]

    fig = go.Figure(go.Bar(
        x=labels,
        y=values,
        marker_color=colors,
        text=values,
        textposition="outside",  # 막대 위에 숫자 표시
        hovertemplate="%{x}: %{y}개<extra></extra>",
    ))
    fig.update_layout(
        yaxis=dict(title="글자 수", tickformat="d", range=[0, max(values) + 1.5]),
        xaxis=dict(tickfont=dict(size=13)),
        margin=dict(t=20, b=20, l=30, r=20),
        height=260,
        showlegend=False,
    )
    return fig


def format_ohang_summary(ohang_dict: dict[str, int], ohang_vector: list[float]) -> str:
    """
    오행 분포를 "목 2 (25.0%) | 화 1 (12.5%) | ..." 형태로 반환.
    Streamlit st.caption 등에 사용.
    """
    parts = []
    for o, v in zip(OHANG_ORDER, ohang_vector):
        count = ohang_dict.get(o, 0)
        parts.append(f"{o} {count} ({v*100:.1f}%)")
    return " | ".join(parts)
