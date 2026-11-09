"""
BankSim(bs140513_032310.csv) 전처리 -- 2026-09 팀 회의 확정 "데이터 전처리 가이드" 구현

원본 컬럼: step, customer, age, gender, zipcodeOri, merchant, zipMerchant, category, amount, fraud
  -> 이번 라운드에서는 age/gender/zipcodeOri/zipMerchant 는 사용하지 않음(팀 결정).
     사용 컬럼: customer, step, merchant, category, amount, fraud

1단계 정제: 결측 amount 행 제거, 결측 category는 최빈값으로 대체,
           amount는 상하위 0.1%(train 기준)로 winsorize, (customer,step,amount) 중복 제거
2단계 인코딩: category -> 정수 ID(train 기준), dow=step%7 -> sin/cos cyclical encoding
3단계 파생: 고객별 시간순 정렬 후 delta_t/mean_5/std_5/velocity/category_diversity_5 계산,
           log_amount=log(amount+1) 계산, (log_amount, delta_t, mean_5, std_5) train 기준 z-score 정규화
4단계 시퀀스: L=5/10/20 각각에 대해 zero-padding + is_padding 마스크로 시퀀스 생성
5단계 분할: 절대 시간(step) 기준 70/15/15, fraud 비율 점검 후 필요시 stratified temporal split 보정

출력:
  transaction_level.csv         : 거래 1건 = 1행 (MLP용), split 컬럼 포함 통합 테이블
  seq_L{5,10,20}/{train,val,test}.npz : 시퀀스 모델(LSTM/GRU/Transformer/TCN)용
  seq_L5/{train,val,test}.csv   : L=5 시퀀스를 long format으로 펼친 예시(파일 크기상 L=5만 CSV 제공)
"""
import argparse
import os
import numpy as np
import pandas as pd

from common_utils import (
    chronological_split_by_global_time, assign_split_column, stratified_temporal_split,
    check_fraud_ratio, print_fraud_ratio_report,
    compute_winsorize_bounds, apply_winsorize, dedup_by_keys,
    fit_category_mapping, apply_category_mapping, dow_cyclical_encoding,
    ZScoreScaler, compute_causal_derived_features,
    build_padded_sequences, print_split_report,
)

K = 5  # mean_k/std_k/velocity/category_diversity_k 의 k (가이드에서 고정값)
SEQ_LENS = [5, 10, 20]
PREVIEW_N = 200  # L=5 long-format CSV 미리보기에 포함할 샘플(시퀀스) 개수


def load_raw(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    str_cols = ["customer", "merchant", "category"]
    for c in str_cols:
        df[c] = df[c].astype(str).str.strip("'")
    df = df[["customer", "step", "merchant", "category", "amount", "fraud"]].copy()
    df["step"] = df["step"].astype(np.int64)
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["fraud"] = df["fraud"].astype(np.float32)
    return df


def step1_clean(df: pd.DataFrame):
    n0 = len(df)
    df = df.dropna(subset=["amount"]).copy()
    n1 = len(df)
    print(f"[1단계-정제] 결측 amount 제거: {n0 - n1:,}건 삭제 (남은 {n1:,}건)")

    if df["category"].isna().any():
        mode_cat = df["category"].mode(dropna=True).iloc[0]
        n_missing = df["category"].isna().sum()
        df["category"] = df["category"].fillna(mode_cat)
        print(f"[1단계-정제] 결측 category {n_missing:,}건을 최빈값 '{mode_cat}'로 대체")
    else:
        print("[1단계-정제] category 결측치 없음")

    df["customer_id"] = "BS_" + df["customer"]
    df, n_dup = dedup_by_keys(df, ["customer_id", "step", "amount"])
    print(f"[1단계-정제] (customer,step,amount) 중복 {n_dup:,}건 제거 (남은 {len(df):,}건)")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="../data/banksim/bs140513_032310.csv")
    ap.add_argument("--out", default="../data/processed/banksim")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print(f"[BankSim] 원본 로드: {args.csv}")
    raw = load_raw(args.csv)
    df = step1_clean(raw)

    # --- split 경계는 정제 완료된 데이터 기준으로 미리 계산 (winsorize/정규화가 train만 봐야 하므로) ---
    train_end, val_end = chronological_split_by_global_time(df, "step", 0.70, 0.15)
    df["split"] = assign_split_column(df, "step", train_end, val_end)
    train_mask = df["split"] == "train"
    print(f"[5단계-분할] 1차 split 경계(step 기준): train<= {train_end}, val<= {val_end}, test> {val_end}")

    # --- 1단계: winsorize (train 기준 상하위 0.1%) ---
    lo, hi = compute_winsorize_bounds(df.loc[train_mask, "amount"].to_numpy(), 0.001, 0.999)
    n_clipped = int(((df["amount"] < lo) | (df["amount"] > hi)).sum())
    df["amount"] = apply_winsorize(df["amount"].to_numpy(), lo, hi)
    print(f"[1단계-정제] amount winsorize 경계(train 0.1%~99.9%): [{lo:.2f}, {hi:.2f}] -> {n_clipped:,}건 clip")

    # --- 2단계: 인코딩 ---
    cat_map, unk_id = fit_category_mapping(df.loc[train_mask, "category"])
    df["category_id"] = apply_category_mapping(df["category"], cat_map, unk_id)
    n_unknown = int((df["category_id"] == unk_id).sum())
    print(f"[2단계-인코딩] category -> 정수 ID {len(cat_map)}개 (train 기준) "
          f"+ unknown_id={unk_id} (val/test 전용 미지 category {n_unknown:,}건)")

    dow, sin_dow, cos_dow = dow_cyclical_encoding(df["step"])
    df["dow"] = dow
    df["sin_dow"] = sin_dow
    df["cos_dow"] = cos_dow
    print("[2단계-인코딩] dow=step%7 -> sin_dow/cos_dow cyclical encoding 완료 "
          "(step에 시각 정보가 없어 hour-of-day 인코딩은 불가)")

    # --- 3단계: 고객별 정렬 + 파생 feature ---
    df = df.sort_values(["customer_id", "step"], kind="mergesort").reset_index(drop=True)
    # split은 정렬 후에도 그대로 유지되지만, groupby 기준값이 바뀌지 않았으므로 재계산 불필요
    train_mask = df["split"] == "train"

    delta_t, mean_k, std_k, velocity, cat_div = compute_causal_derived_features(
        df, "customer_id", "step", "amount", "category_id", k=K
    )
    df["delta_t_raw"] = delta_t
    df[f"mean_{K}_raw"] = mean_k
    df[f"std_{K}_raw"] = std_k
    df["velocity"] = velocity
    df[f"category_diversity_{K}"] = cat_div
    df["log_amount"] = np.log1p(df["amount"].clip(lower=0))
    print(f"[3단계-파생] delta_t / mean_{K} / std_{K} / velocity / category_diversity_{K} / log_amount 계산 완료")

    # train 기준 z-score fit -> 전체에 transform (컬럼 자체를 정규화된 값으로 덮어씀, 가이드 스키마와 동일)
    scalers = {}
    for raw_col, out_col in [
        ("log_amount", "amount_zscore"),
        ("delta_t_raw", "delta_t"),
        (f"mean_{K}_raw", f"mean_{K}"),
        (f"std_{K}_raw", f"std_{K}"),
    ]:
        sc = ZScoreScaler().fit(df.loc[train_mask, raw_col].to_numpy())
        df[out_col] = sc.transform(df[raw_col].to_numpy())
        scalers[out_col] = sc
        print(f"[3단계-정규화] {raw_col} -> {out_col} : train mean={sc.mean_:.4f}, std={sc.std_:.4f}")

    # --- 5단계: fraud 비율 점검, 필요시 stratified temporal split으로 보정 ---
    ok, overall_rate, report = check_fraud_ratio(df, "split", "fraud", tol=0.30)
    print_fraud_ratio_report("1차 split(절대 시간)", overall_rate, report)
    if not ok:
        print("[5단계-분할] fraud 비율 편차가 커서 stratified temporal split으로 재분할합니다.")
        df["split"] = stratified_temporal_split(df, "step", "fraud", 0.70, 0.15)
        ok2, overall_rate2, report2 = check_fraud_ratio(df, "split", "fraud", tol=0.30)
        print_fraud_ratio_report("2차 split(stratified temporal)", overall_rate2, report2)
    else:
        print("[5단계-분할] fraud 비율이 허용 범위 내 -> 절대 시간 기준 split 그대로 사용")

    # --- 최종 transaction-level 테이블 ---
    final_cols = {
        "customer_id": df["customer_id"],
        "step": df["step"],
        "merchant_id": df["merchant"],
        "category": df["category_id"],
        "amount": df["amount"],
        "log_amount": df["log_amount"],
        "amount_zscore": df["amount_zscore"],
        "sin_dow": df["sin_dow"],
        "cos_dow": df["cos_dow"],
        "delta_t": df["delta_t"],
        f"mean_{K}": df[f"mean_{K}"],
        f"std_{K}": df[f"std_{K}"],
        "velocity": df["velocity"],
        f"category_diversity_{K}": df[f"category_diversity_{K}"],
        "fraud": df["fraud"].astype(np.int64),
        "split": df["split"],
    }
    final_df = pd.DataFrame(final_cols)

    # 통합본은 로컬 확인용으로만 저장(105MB 정도로 GitHub 100MB 하드리밋을 살짝 넘김).
    tx_csv_path = os.path.join(args.out, "transaction_level.csv")
    final_df.to_csv(tx_csv_path, index=False)
    size_mb = os.path.getsize(tx_csv_path) / 1e6
    print(f"[출력] transaction_level.csv(통합, 로컬 확인용) : {len(final_df):,}행 x {final_df.shape[1]}컬럼 "
          f"-> {tx_csv_path} ({size_mb:.1f} MB)")

    # git 업로드용으로는 split별로 나눠서 저장 (각각 100MB 미만, split 컬럼은 그대로 유지)
    for split_name in ["train", "val", "test"]:
        split_path = os.path.join(args.out, f"transaction_level_{split_name}.csv")
        sub = final_df[final_df["split"] == split_name]
        sub.to_csv(split_path, index=False)
        size_mb = os.path.getsize(split_path) / 1e6
        print(f"[출력] transaction_level_{split_name}.csv : {len(sub):,}행 -> {split_path} ({size_mb:.1f} MB)")

    # --- 4단계: 시퀀스(L=5/10/20) ---
    seq_feature_cols = [
        "category", "amount_zscore", "sin_dow", "cos_dow",
        "delta_t", f"mean_{K}", f"std_{K}", "velocity", f"category_diversity_{K}",
    ]
    seq_source = final_df.copy()
    seq_source["fraud"] = seq_source["fraud"].astype(np.float32)

    for L in SEQ_LENS:
        seq_dir = os.path.join(args.out, f"seq_L{L}")
        os.makedirs(seq_dir, exist_ok=True)
        print(f"[4단계-시퀀스] L={L} 생성 중...")
        for split_name in ["train", "val", "test"]:
            sub = seq_source[seq_source["split"] == split_name].sort_values(
                ["customer_id", "step"], kind="mergesort"
            ).reset_index(drop=True)
            X, is_padding, y, cust = build_padded_sequences(
                sub, seq_feature_cols, "fraud", "customer_id", seq_len=L
            )
            print_split_report(f"L={L}/{split_name}", y)
            np.savez_compressed(
                os.path.join(seq_dir, f"{split_name}.npz"),
                X=X, is_padding=is_padding, y=y, customer_id=cust,
                feature_cols=np.array(seq_feature_cols),
            )
            if L == 5:
                # 전체를 long-format CSV로 펼치면 수백 MB(예: train 213MB)라 git에 올리기엔 너무 큼.
                # 실제 학습 데이터는 .npz(위, 수 MB)로 충분하므로, CSV는 형식 확인용 미리보기
                # PREVIEW_N개 샘플만 long format(가이드 예시와 동일한 컬럼 구조)으로 남긴다.
                n, seq_len, n_feat = X.shape
                n_preview = min(PREVIEW_N, n)
                Xp, ip, yp, cp = X[:n_preview], is_padding[:n_preview], y[:n_preview], cust[:n_preview]
                lag_labels = [f"t-{seq_len - 1 - p}" if p < seq_len - 1 else "t" for p in range(seq_len)]
                sample_id = np.repeat(np.arange(n_preview), seq_len)
                seq_position = np.tile(lag_labels, n_preview)
                cust_long = np.repeat(cp, seq_len)
                ispad_long = ip.reshape(-1)
                feat_long = Xp.reshape(n_preview * seq_len, n_feat)
                long_df = pd.DataFrame(feat_long, columns=seq_feature_cols)
                long_df.insert(0, "seq_position", seq_position)
                long_df.insert(0, "sample_id", sample_id)
                long_df.insert(2, "customer_id", cust_long)
                long_df["is_padding"] = ispad_long
                fraud_col = np.full(n_preview * seq_len, np.nan)
                fraud_col[seq_len - 1::seq_len] = yp  # 마지막(t) 위치에만 라벨 부착
                long_df["fraud"] = fraud_col
                csv_path = os.path.join(seq_dir, f"{split_name}_preview.csv")
                long_df.to_csv(csv_path, index=False)
                size_mb = os.path.getsize(csv_path) / 1e6
                print(f"    -> {csv_path} (미리보기 {n_preview}/{n:,}개 샘플, {len(long_df):,}행, {size_mb:.2f} MB)")

    print(f"[BankSim] 전처리 완료. 저장 위치: {args.out}")
    print(f"[BankSim] 시퀀스 feature 순서: {seq_feature_cols}")


if __name__ == "__main__":
    main()
