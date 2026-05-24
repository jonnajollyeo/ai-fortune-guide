# 🔮 AI 운명 가이드

오행 에너지 분석 · 연예인 매칭 · AI 맞춤 사주 해설 플랫폼

## 주요 기능

- **오행 에너지 분석** — 생년월일시로 5차원 벡터 산출, 레이더·막대 차트 시각화
- **연예인 매칭** — 코사인 유사도로 에너지가 비슷한 연예인 TOP 3 매칭 (K-Pop 아이돌 1,800명 DB)
- **AI 맞춤 해설** — Gemini 2.5 Flash 기반 페르소나 상담 (냉철한 선비 / 따뜻한 조언가 / MZ 술사)
- **추가 질문** — 채팅 형식으로 최대 5회 후속 상담
- **리포트 다운로드** — 결과를 마크다운 파일로 저장

## 로컬 실행

```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. API 키 설정
echo "GEMINI_API_KEY=your_key_here" > .env

# 3. 앱 실행
streamlit run app.py
```

Gemini API 키는 [Google AI Studio](https://aistudio.google.com/app/apikey)에서 발급받을 수 있습니다.

## Streamlit Cloud 배포

1. 이 저장소를 GitHub에 push
2. [share.streamlit.io](https://share.streamlit.io) → "New app" → 저장소 선택, `app.py` 지정
3. **Advanced settings → Secrets**에 아래 내용 입력:

```toml
GEMINI_API_KEY = "your_key_here"
```

## 기술 스택

| 영역 | 기술 |
|------|------|
| 프론트엔드 | Streamlit |
| AI 해설 | Gemini 2.5 Flash (google-genai) |
| 만세력 계산 | sxtwl |
| 유사도 분석 | scikit-learn (cosine similarity) |
| 시각화 | Plotly |
| 데이터 | Kaggle K-Pop Idols DB (1,798명) |
