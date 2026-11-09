"""검증 F1로 임계값 τ를 고르고 test에 적용한다. test는 τ 선택에 쓰지 않는다.
사용: python threshold_sweep.py hparam_results/probs_tcn_L5_s42.npz"""
import sys
import numpy as np
from sklearn.metrics import precision_recall_fscore_support as prf

d = np.load(sys.argv[1])
pv, yv, pt, yt = d["pv"], d["yv"], d["pt"], d["yt"]
grid = np.round(np.arange(0.90, 0.9951, 0.005), 3)
best_t, best_f = None, -1
for t in grid:
    f = prf(yv, pv >= t, average="binary", zero_division=0)[2]
    if f > best_f:
        best_t, best_f = t, f
p, r, f1, _ = prf(yt, pt >= best_t, average="binary", zero_division=0)
print(f"tau*={best_t} val_F1={best_f:.4f} | test P={p:.4f} R={r:.4f} F1={f1:.4f}")
if best_t == grid[-1]:
    print("주의: 최적값이 탐색 범위의 끝입니다.")
