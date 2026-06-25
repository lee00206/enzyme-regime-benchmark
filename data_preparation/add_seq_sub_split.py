import pandas as pd
import numpy as np

CSV_PATH = "/disk1/AI_Models/KcatPrediction/KcatPaper/data/combined_brenda_sabiork_geometric_mean_split.csv"

PH_LO,   PH_HI   = 6.0,  9.0
TEMP_LO, TEMP_HI = 22.0, 60.0

SPLIT_MAP = {
    "80_split": "80_seq_sub_split",
    "60_split": "60_seq_sub_split",
    "40_split": "40_seq_sub_split",
}


def score(pH, temp):
    ph_miss   = (pH   == -1) or pd.isna(pH)
    temp_miss = (temp == -1) or pd.isna(temp)

    pH_in   = (not ph_miss)   and (PH_LO   < pH   <= PH_HI)
    temp_in = (not temp_miss) and (TEMP_LO < temp <= TEMP_HI)

    if pH_in and temp_in:
        return 0, 0.0
    if pH_in or temp_in:
        return 1, 0.0

    ph_dist = (9999.0 if ph_miss
               else PH_LO - pH   if pH <= PH_LO
               else pH - PH_HI   if pH > PH_HI
               else 0.0)
    temp_dist = (9999.0 if temp_miss
                 else TEMP_LO - temp if temp <= TEMP_LO
                 else temp - TEMP_HI if temp > TEMP_HI
                 else 0.0)
    return 2, ph_dist + temp_dist


def apply_seq_sub_split(df: pd.DataFrame, split_col: str, out_col: str) -> pd.Series:
    tmp = df[[split_col, "sequence", "canonical_smiles", "pH", "temperature"]].copy()

    scores = tmp.apply(lambda r: score(r["pH"], r["temperature"]),
                       axis=1, result_type="expand")
    tmp["_priority"] = scores[0]
    tmp["_distance"] = scores[1]

    best_indices = set(
        tmp.sort_values(["_priority", "_distance"])
           .groupby([split_col, "sequence", "canonical_smiles"])
           .apply(lambda g: g.index[0], include_groups=False)
           .values
    )

    return df.apply(
        lambda r: r[split_col] if r.name in best_indices else "omit",
        axis=1,
    )


def main():
    print(f"Loading: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    print(f"  Rows: {len(df)}")

    for split_col, out_col in SPLIT_MAP.items():
        print(f"\n=== {split_col} → {out_col} ===")

        df[out_col] = apply_seq_sub_split(df, split_col, out_col)

        counts = df[out_col].value_counts()
        total = len(df)
        selected = (df[out_col] != "omit").sum()
        omitted  = (df[out_col] == "omit").sum()
        print(f"  train : {counts.get('train', 0):>6}")
        print(f"  valid : {counts.get('valid', 0):>6}")
        print(f"  test  : {counts.get('test',  0):>6}")
        print(f"  omit  : {omitted:>6}")
        print(f"  Selected : {selected} / {total}")

        # check leakage
        sel = df[df[out_col] != "omit"]
        test_pairs = set(zip(
            sel[sel[out_col] == "test"]["sequence"],
            sel[sel[out_col] == "test"]["canonical_smiles"],
        ))
        tv_pairs = set(zip(
            sel[sel[out_col].isin(["train", "valid"])]["sequence"],
            sel[sel[out_col].isin(["train", "valid"])]["canonical_smiles"],
        ))
        leak = len(test_pairs & tv_pairs)
        print(f"  Test ↔ Train/Valid (seq, smiles) leakage: {leak}개 (must be 0)")

    print(f"\nSaving: {CSV_PATH}")
    df.to_csv(CSV_PATH, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
