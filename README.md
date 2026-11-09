# 계은혜 · 데이터 프로젝트 모음

동덕여자대학교 데이터사이언스전공 3학년 계은혜의 수업 팀 프로젝트 코드입니다.
세 프로젝트 모두 Python으로 데이터를 정리하고, 모델을 돌리거나 시각화한 결과를 담았습니다.

| | 프로젝트 | 수업 · 기간 | 내 역할 | 핵심 결과 |
|---|---|---|---|---|
| 01 | [개별 거래 모델과 시퀀스 딥러닝 모델의 금융 사기 탐지 성능 비교](01_deep_learning_fds) | 딥러닝 · 2026-2 · 3인 | 조장 · TCN·Transformer 구현, 결과 시각화 | TCN(L=5) PR-AUC 0.8707로 6개 모델 중 1위, 판단 기준 조정으로 F1 0.51 → 0.80 |
| 02 | [서울시 종로구·성북구 CCTV·보안등 공간 밀집도 분석](02_dataviz_cctv) | 데이터시각화이해와실습 · 2026-1 · 5인 | 데이터 전처리·시각화 코드, 회의록 | 종로구 이중 취약 지역 7개 동 도출, 시각화 6종 × 2개 구, A+ |
| 03 | [KOD-FedGNN 기반 신규 사용자 음악 추천](03_ml_music_recsys) | 기계학습 · 2026-1 · 5인 | 조장 · 추천 파이프라인 코드 | K-means 군집 → 온보딩 10곡 → 의사결정나무 → 점수 합산 Top-10 추천 파이프라인 구현 |

## 사용한 도구

- **데이터 처리:** pandas, NumPy
- **모델:** PyTorch (GRU·LSTM·Transformer·TCN), scikit-learn (K-means, 의사결정나무)
- **시각화:** matplotlib, seaborn, folium (단계구분도·히트맵·산점도 지도)
- **환경:** VS Code, Jupyter, Google Colab, GitHub

## 폴더 구조

```
01_deep_learning_fds/   BankSim 금융 사기 탐지 (PyTorch)
02_dataviz_cctv/        종로구·성북구 CCTV·보안등 시각화 (pandas · folium)
03_ml_music_recsys/     신규 사용자 음악 추천 파이프라인 (scikit-learn)
```

원본 데이터(CSV)는 용량과 배포 조건 때문에 올리지 않았습니다. 받는 곳은 각 폴더 README에 적었습니다.

연락처: geh1210@naver.com
