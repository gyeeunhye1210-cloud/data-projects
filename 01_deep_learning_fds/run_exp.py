"""기존 hparam_search.py의 함수(train_one 등)를 그대로 import해서 사용. seed만 바꿈."""
import sys, json, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hparam_search as hs
import torch, numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, precision_score, recall_score

device = torch.device('cpu')
torch.set_num_threads(2)
cfgs = json.load(open('hparam_results/best_config.json'))

def run(name, L, seed):
    hs.SEED = seed          # train_one 내부 set_seed(SEED), Generator.manual_seed(SEED) 모두 이 전역을 읽음
    tag = f"{name}_L{L}_s{seed}"
    cfg = {k: cfgs[name][k] for k in ("lr", "hidden", "dropout")}
    res = hs.train_one(name, L, cfg, hs.FINAL_MAX_EPOCHS, hs.FINAL_PATIENCE, device, True, tag)
    # 저장된 best_state로 val/test 확률 재계산 (실험 A용)
    data = hs.load_L(L)
    model = hs.build_model(name, 9, cfg["hidden"], cfg["dropout"])
    model.load_state_dict(torch.load(f"{hs.OUT_DIR}/final_{tag}.pt"))
    out = {}
    for sp in ("val", "test"):
        dl = DataLoader(TensorDataset(*data[sp]), batch_size=hs.BATCH_SIZE)
        _, p, y = hs.run_epoch(model, dl, None, torch.nn.BCEWithLogitsLoss(), device, False)
        out[sp] = (p, y)
    np.savez(f"{hs.OUT_DIR}/probs_{tag}.npz", pv=out['val'][0], yv=out['val'][1], pt=out['test'][0], yt=out['test'][1])
    json.dump(res, open(f"{hs.OUT_DIR}/res_{tag}.json", "w"), indent=1)
    return res

if __name__ == "__main__":
    jobs = [(a[0], int(a[1]), int(a[2])) for a in (x.split(':') for x in sys.argv[1:])]
    for n, L, s in jobs:
        if os.path.exists(f"{hs.OUT_DIR}/res_{n}_L{L}_s{s}.json"):
            continue
        run(n, L, s)
