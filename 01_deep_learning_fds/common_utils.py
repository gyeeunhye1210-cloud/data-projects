"""
BankSim / IBM(sd254) 공통 전처리 유틸리티
-- 2026-09 팀 회의에서 확정한 "데이터 전처리 가이드" 기준으로 재작성됨 --

5단계 파이프라인 (가이드 그대로):
  1. 정제(Cleaning): 결측치 처리, amount winsorize(상하위 0.1%, train 기준 경계값),
     (customer, step, amount) 동일 행 중복 제거
  2. 인코딩(Encoding): category -> 정수 ID, 요일(dow) cyclical encoding(sin/cos)
  3. 고객별 정렬 + 파생 feature(delta_t/mean_k/std_k/velocity/category_diversity_k)
     계산 + 정규화(z-score, train 기준 mu/sigma)
  4. 시퀀스(윈도우) 구성: L=5/10/20, 과거 이력이 L개 미만이면 zero-padding + is_padding 마스크
  5. 분할: 절대 시간 기준 70/15/15 (셔플 없음), split별 fraud 비율 점검 후 필요시
     stratified temporal split으로 보정

주의: 여기서 만드는 모든 파생 feature는 "현재 시점 이전"의 값만 사용하는 causal feature이므로
(미래 거래를 들여다보지 않으므로) train/val/test 전체 고객 이력에 대해 한 번에 계산해도
데이터 누출이 아니다. 다만 정규화에 쓰는 평균/표준편차, winsorize 경계값, category id 매핑처럼
"통계량을 추정"하는 부분은 반드시 train 구간 데이터로만 fit 해야 한다 (아래 함수들이 이 원칙을 따름).
"""
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 0단계: 절대 시간 기준 70/15/15 split 경계
# ---------------------------------------------------------------------------
def chronological_split_by_global_time(df: pd.DataFrame, time_col: str, train_frac=0.70, val_frac=0.15):
    """
    전체 타임라인을 기준으로 train/val/test 경계값(시간 값 자체)을 계산한다.
    행 개수가 아니라 '시간 값'을 기준으로 정렬 후 percentile 위치의 값을 경계로 삼기 때문에,
    같은 시간(step) 값을 가진 행들이 서로 다른 split으로 쪼개지지 않는다(같은 날 거래는 항상 같은 split).
    """
    t_sorted = np.sort(df[time_col].to_numpy())
    n = len(t_sorted)
    train_end = t_sorted[max(int(n * train_frac) - 1, 0)]
    val_end = t_sorted[max(int(n * (train_frac + val_frac)) - 1, 0)]
    return train_end, val_end


def assign_split_column(df: pd.DataFrame, time_col: str, train_end, val_end) -> pd.Series:
    t = df[time_col].to_numpy()
    split = np.where(t <= train_end, "train", np.where(t <= val_end, "val", "test"))
    return pd.Series(split, index=df.index, name="split")


def stratified_temporal_split(df: pd.DataFrame, time_col: str, label_col: str,
                               train_frac=0.70, val_frac=0.15) -> pd.Series:
    """
    fraud 비율이 split마다 너무 벌어질 때 쓰는 보정판.
    정상 거래(0)와 사기 거래(1)를 각각 따로 시간순 70/15/15로 나눈 뒤 합친다.
    -> 두 그룹 모두 내부적으로는 여전히 "과거로 학습 -> 미래로 검증" 순서를 지키면서,
       각 split의 fraud 비율은 원본 전체 비율에 훨씬 가깝게 유지된다.
    """
    split = pd.Series(index=df.index, dtype=object)
    for _, g in df.groupby(label_col):
        train_end, val_end = chronological_split_by_global_time(g, time_col, train_frac, val_frac)
        s = assign_split_column(g, time_col, train_end, val_end)
        split.loc[g.index] = s
    return split


def check_fraud_ratio(df: pd.DataFrame, split_col: str, label_col: str, tol=0.30):
    """
    split별 fraud 비율이 전체 비율 대비 상대오차 tol(기본 30%)을 넘는지 점검.
    반환: (ok: bool, report: dict[split] = {n, fraud, rate})
    """
    overall_rate = df[label_col].mean()
    ok = True
    report = {}
    for name, g in df.groupby(split_col):
        rate = g[label_col].mean()
        rel_dev = abs(rate - overall_rate) / overall_rate if overall_rate > 0 else 0.0
        report[name] = {"n": len(g), "fraud": int(g[label_col].sum()), "rate": rate, "rel_dev": rel_dev}
        if rel_dev > tol:
            ok = False
    return ok, overall_rate, report


def print_fraud_ratio_report(tag: str, overall_rate: float, report: dict):
    print(f"  [{tag}] 전체 fraud 비율={overall_rate*100:.4f}%")
    for name in ("train", "val", "test"):
        if name in report:
            r = report[name]
            print(f"    - {name}: n={r['n']:,} fraud={r['fraud']:,} rate={r['rate']*100:.4f}% "
                  f"(상대오차 {r['rel_dev']*100:.1f}%)")


# ---------------------------------------------------------------------------
# 1단계: 정제 (winsorize / 중복 제거)
# ---------------------------------------------------------------------------
def compute_winsorize_bounds(train_values: np.ndarray, lower_pct=0.001, upper_pct=0.999):
    v = np.asarray(train_values, dtype=np.float64)
    v = v[~np.isnan(v)]
    lo = np.percentile(v, lower_pct * 100)
    hi = np.percentile(v, upper_pct * 100)
    return float(lo), float(hi)


def apply_winsorize(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=np.float64), lo, hi)


def dedup_by_keys(df: pd.DataFrame, keys: list):
    before = len(df)
    out = df.drop_duplicates(subset=keys, keep="first").reset_index(drop=True)
    return out, before - len(out)


# ---------------------------------------------------------------------------
# 2단계: 인코딩 (category -> 정수 ID / 요일 cyclical encoding)
# ---------------------------------------------------------------------------
def fit_category_mapping(train_categories: pd.Series):
    """
    train 구간에 등장한 category만으로 정수 ID를 매긴다(0, 1, 2, ...).
    val/test에만 등장하는 미지의 category는 마지막 번호(unknown_id)로 몰아준다.
    """
    uniq = sorted(train_categories.dropna().astype(str).unique().tolist())
    mapping = {cat: i for i, cat in enumerate(uniq)}
    unknown_id = len(uniq)
    return mapping, unknown_id


def apply_category_mapping(series: pd.Series, mapping: dict, unknown_id: int) -> np.ndarray:
    return series.astype(str).map(lambda v: mapping.get(v, unknown_id)).astype(np.int64).to_numpy()


def cyclical_encode(dow: np.ndarray):
    """dow(0~6) 배열을 sin/cos로 cyclical encoding."""
    dow = np.asarray(dow, dtype=np.int64)
    angle = 2 * np.pi * dow / 7.0
    return np.sin(angle), np.cos(angle)


def dow_cyclical_encoding(step: pd.Series):
    """
    BankSim 전용: dow = step % 7 로 요일을 근사한다
    (step이 날짜 카운터일 뿐 실제 달력 정보가 없어 나온 근사치이며, 시간대(hour-of-day) 인코딩은 불가능).
    IBM처럼 실제 달력 날짜가 있는 경우는 이 함수 대신 실제 요일을 구해서 cyclical_encode()에 넘길 것.
    """
    dow = (step.astype(np.int64) % 7).to_numpy()
    sin_dow, cos_dow = cyclical_encode(dow)
    return dow, sin_dow, cos_dow


# ---------------------------------------------------------------------------
# 3단계: 표준화(z-score, train 기준 fit)
# ---------------------------------------------------------------------------
class ZScoreScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float64)
        self.mean_ = np.nanmean(x)
        self.std_ = np.nanstd(x)
        if self.std_ == 0 or np.isnan(self.std_):
            self.std_ = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        x = np.nan_to_num(x, nan=self.mean_)
        return (x - self.mean_) / self.std_

    def fit_transform(self, x):
        return self.fit(x).transform(x)


# ---------------------------------------------------------------------------
# 3단계: 고객별 causal 파생 feature
#   - 반드시 (customer_col, time_col) 오름차순 정렬된 df에 대해 호출할 것
#   - 각 거래 시점에서 "자기 자신을 제외한" 직전 최대 k개 거래만 사용 (미래 누출 없음)
#   - 첫 거래(이전 이력 0개)는 delta_t=0, mean_k=0, std_k=0, velocity=0, category_diversity_k=0
# ---------------------------------------------------------------------------
def compute_causal_derived_features(df: pd.DataFrame, customer_col: str, time_col: str,
                                     amount_col: str, category_col: str, k: int = 5):
    g = df.groupby(customer_col, sort=False)

    # delta_t: 직전 거래와의 step 차이 (첫 거래는 0)
    delta_t = g[time_col].diff().fillna(0.0).to_numpy(dtype=np.float64)

    # amount의 shift(1..k) -> "현재 제외, 직전 최대 k개"
    amt_lag_cols = [g[amount_col].shift(lag) for lag in range(1, k + 1)]
    amt_lags = pd.concat(amt_lag_cols, axis=1).to_numpy(dtype=np.float64)  # (n, k)

    valid = ~np.isnan(amt_lags)
    velocity = valid.sum(axis=1).astype(np.int64)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_k = np.nanmean(amt_lags, axis=1)
        std_k = np.nanstd(amt_lags, axis=1)
    mean_k = np.nan_to_num(mean_k, nan=0.0)
    std_k = np.nan_to_num(std_k, nan=0.0)

    # category diversity: shift(1..k) 값들 중 서로 다른 값 개수 (k가 작으므로 O(k^2) 비교로 충분히 빠름)
    cat_lag_cols = [g[category_col].shift(lag) for lag in range(1, k + 1)]
    cat_lags = pd.concat(cat_lag_cols, axis=1).to_numpy(dtype=np.float64)  # NaN 포함 가능하므로 float
    cat_valid = ~np.isnan(cat_lags)
    n = cat_lags.shape[0]
    diversity = np.zeros(n, dtype=np.int64)
    for i in range(k):
        is_new = cat_valid[:, i].copy()
        for j in range(i):
            same = cat_valid[:, j] & (cat_lags[:, i] == cat_lags[:, j])
            is_new &= ~same
        diversity += is_new.astype(np.int64)

    return delta_t, mean_k, std_k, velocity, diversity


# ---------------------------------------------------------------------------
# 4단계: 시퀀스(윈도우) 구성 -- zero-padding + is_padding 마스크
#   - include_current=True 방식으로 고정: [t-L+1 ... t] 를 하나의 샘플로 묶고,
#     라벨은 가장 최근 시점(t, 현재 거래) 하나의 fraud 값만 사용한다.
#   - 과거 이력이 L개 미만인 고객(신규 고객 등)은 건너뛰지 않고 앞쪽을 0으로 zero-padding한다.
#     -> 모든 거래가 하나씩의 시퀀스 샘플이 되며, padding 위치는 is_padding=1로 표시된다.
#     -> 학습 시 loss/attention 계산에서 is_padding=1인 위치는 반드시 제외해야 한다.
# ---------------------------------------------------------------------------
def build_padded_sequences(df: pd.DataFrame, feature_cols: list, label_col: str, customer_col: str,
                            seq_len: int):
    feats = df[feature_cols].to_numpy(dtype=np.float32)
    labels = df[label_col].to_numpy(dtype=np.float32)
    customers = df[customer_col].to_numpy()
    n, n_feat = feats.shape

    X = np.zeros((n, seq_len, n_feat), dtype=np.float32)
    is_padding = np.ones((n, seq_len), dtype=np.int8)

    change_points = np.where(customers[1:] != customers[:-1])[0] + 1
    boundaries = np.concatenate(([0], change_points, [n]))

    for start, end in zip(boundaries[:-1], boundaries[1:]):
        m = end - start
        seg = feats[start:end]
        pad_block = np.zeros((seq_len - 1, n_feat), dtype=np.float32)
        seg_padded = np.concatenate([pad_block, seg], axis=0)  # (m + seq_len - 1, n_feat)
        windows = np.lib.stride_tricks.sliding_window_view(seg_padded, seq_len, axis=0)  # (m, n_feat, seq_len)
        X[start:end] = np.moveaxis(windows, -1, 1)  # -> (m, seq_len, n_feat)

        pos = np.arange(m)
        real_len = np.minimum(seq_len, pos + 1)
        pad_len = seq_len - real_len
        col_idx = np.arange(seq_len)[None, :]
        is_padding[start:end] = (col_idx < pad_len[:, None]).astype(np.int8)

    return X, is_padding, labels, customers


def print_split_report(name: str, y: np.ndarray):
    n = len(y)
    n_fraud = int(y.sum())
    rate = n_fraud / n * 100 if n else 0.0
    print(f"  [{name}] samples={n:>8,d}  fraud={n_fraud:>6,d}  fraud_rate={rate:.3f}%")
