# 금융 사기 탐지: 개별 거래 모델 vs 시퀀스 모델 (BankSim)

고객의 **과거 거래 이력을 쓰면 사기 탐지가 실제로 나아지는가**를 확인하기 위해, 6개 모델을 같은 데이터와 같은 조건에서 비교한 딥러닝 Mini Research Project(동덕여자대학교, 2026-2학기, 3인 팀)입니다.
새 모델을 제안하지 않고, 통제된 조건에서 비교하는 것이 목적입니다.

- **논문 제목:** 개별 거래 모델과 시퀀스 딥러닝 모델의 금융 사기 탐지 성능 비교 — BankSim 데이터를 중심으로
- **내 역할:** 조장 · TCN·Transformer 구현, 결과 시각화, 코드와 README 정리

## 핵심 결과

| 질문 | 결과 |
|---|---|
| 이력 정보가 도움이 되는가 | 이력 없는 MLP 대비 집계 이력 feature를 더한 MLP의 PR-AUC가 0.7347 → 0.8180 |
| 시퀀스로 직접 학습하면 더 좋은가 | TCN(L=5)이 0.8707(GPU, seed 42)로 가장 높음 |
| 결과가 안정적인가 | seed 3개 반복: TCN 0.8621 ± 0.0010, GRU 0.8496 ± 0.0089, LSTM 0.8357 ± 0.0213 (CPU, 표본표준편차) |
| 낮은 Precision의 원인 | 가중 손실 + 고정 임계값 0.5. 검증 F1로 고른 τ*=0.975를 쓰면 test F1 0.5141 → 0.7984 |

- 세 seed 모두 PR-AUC 1위는 TCN(L=5)입니다. 다만 GRU와의 평균 차이(0.013)는 GRU의 seed 간 변동과 비슷해서 우열을 단정하지 않습니다.
- 같은 seed 42라도 GPU와 CPU 결과가 달랐습니다(예: LSTM L=20 PR-AUC 0.8481 대 0.8148). 그래서 반복 실험은 세 seed 모두 CPU에서 학습했습니다.
- 모든 결과는 BankSim 단일 합성 데이터와 이 실험 조건에서의 결과입니다.

## 실험 설정

- 데이터: BankSim 594,643건(사기 1.21%). 정제 후 594,642건.
- 분할: 거래 시점(step) 기준 시간순 70/15/15, 셔플 없음 (train 417,847 / val 87,702 / test 89,093).
- 입력 feature 9개: category, amount_zscore, sin_dow, cos_dow + 직전 5건 기준 delta_t, mean_5, std_5, velocity, category_diversity_5.
- 시퀀스: 같은 고객의 직전 L건(L = 5, 10, 20). 부족하면 앞쪽 zero-padding.
- 손실: 가중 BCE (`pos_weight` = 정상/사기 = 78.14, train 기준).
- 학습: Adam(weight decay 1e-5), batch 2048, 최대 50 epoch, 검증 PR-AUC 기준 조기 종료(patience 5), best 가중치 사용.
- 하이퍼파라미터 탐색: L=10에서 lr × hidden × dropout 27조합 × 6모델, 검증 PR-AUC로 선택. test는 사용하지 않음.
- 평가: PR-AUC(주 지표), ROC-AUC, Precision, Recall, F1(τ=0.5).

## 데이터 누수 점검

정규화 평균·표준편차, winsorize 경계, 업종 매핑, `pos_weight`는 모두 train에서만 계산합니다. 시퀀스와 집계 feature는 같은 고객의 이전 거래만 씁니다. 하이퍼파라미터 탐색, 조기 종료, 모델 선택, 임계값 선택은 모두 검증 데이터만 사용하고 test는 최종 평가에만 씁니다.
알려진 한계: step이 하루 단위라 같은 날(같은 고객·같은 step) 거래의 선후는 CSV 행 순서에 따릅니다. 해당 행은 train 6.7%, val 8.6%, test 8.9%입니다.

## 폴더 구조

```
preprocess_banksim.py   원본 CSV → 정제, 분할, feature 생성, 시퀀스 npz
common_utils.py         분할, winsorize, 파생 feature, 시퀀스 생성, 정규화
hparam_search.py        하이퍼파라미터 탐색 + 최종 학습 (--stage search | final)
run_exp.py              seed를 바꿔 최종 학습 반복 (예: python run_exp.py tcn:5:1 gru:20:1)
threshold_sweep.py      검증 F1로 임계값 선택 후 test 적용
tcn_transformer/        TCN, Transformer 모델 정의
hparam_results/         선택된 하이퍼파라미터, seed별 결과 CSV, TCN 예측 확률
```

## 실행

```bash
pip install -r requirements.txt
# BankSim 원본(bs140513_032310.csv)은 Kaggle에서 받아 ../data/banksim/ 에 둡니다.
python preprocess_banksim.py --csv ../data/banksim/bs140513_032310.csv --out data/processed/banksim
python hparam_search.py                       # 탐색 + 최종 학습
python run_exp.py tcn:5:1 tcn:5:2 gru:20:1    # seed 반복
python threshold_sweep.py hparam_results/probs_tcn_L5_s42.npz
```

## 한계

BankSim 한 종류의 합성 데이터만 썼고 test 사기 거래는 960건입니다. category를 정수 ID로 넣었고 velocity가 사실상 신규 고객 여부만 구분합니다. 이 점을 고친 재실험, 모든 모델의 임계값 조정, 추론 시간 측정은 하지 못했습니다.

## 참고

- E. A. Lopez-Rojas and S. Axelsson, "BankSim: A bank payments simulator for fraud detection research," EMSS, 2014.
- 팀: 계은혜(조장), 김예린, 박수빈
