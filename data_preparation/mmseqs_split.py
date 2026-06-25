import os
import subprocess
import tempfile
import random
import pandas as pd
from pathlib import Path

MMSEQS = "/disk1/AI_Models/KcatPrediction/KEEN/mmseqs/bin/mmseqs"
CSV_PATH = "/disk1/AI_Models/KcatPrediction/KcatPaper/data/combined_brenda_sabiork_geometric_mean_split.csv"
OUTPUT_PATH = CSV_PATH 

THRESHOLDS = [0.8, 0.6, 0.4]
TRAIN_RATIO = 0.8
VALID_RATIO = 0.1
TEST_RATIO  = 0.1 (remainder)
SEED = 42


def write_fasta(sequences: list[str], fasta_path: str):
    """Write unique sequences to a FASTA file using index as header."""
    with open(fasta_path, "w") as f:
        for i, seq in enumerate(sequences):
            f.write(f">seq{i}\n{seq}\n")


def run_mmseqs_cluster(fasta_path: str, out_prefix: str, tmp_dir: str, min_seq_id: float) -> str:
    """Run MMseqs2 easy-cluster and return path to cluster TSV."""
    cmd = [
        MMSEQS, "easy-cluster",
        fasta_path,
        out_prefix,
        tmp_dir,
        "--min-seq-id", str(min_seq_id),
        "-c", "0.8",
        "--cov-mode", "0",
        "--cluster-mode", "0",
        "--threads", "8",
        "-v", "1",
    ]
    print(f"  Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    tsv_path = out_prefix + "_cluster.tsv"
    assert os.path.exists(tsv_path), f"Expected cluster TSV not found: {tsv_path}"
    return tsv_path


def parse_clusters(tsv_path: str) -> dict[str, list[str]]:
    """Parse MMseqs2 cluster TSV → {representative: [members...]}."""
    clusters: dict[str, list[str]] = {}
    with open(tsv_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            rep, member = parts[0], parts[1]
            clusters.setdefault(rep, []).append(member)
    return clusters


def assign_train_valid_test(
    clusters: dict[str, list[str]],
    header_counts: dict[str, int],
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, str]:
    """
    Randomly assign whole clusters to train/valid/test based on the actual
    number of dataframe rows each cluster represents.
    Returns {seq_header: split_label}.
    """
    random.seed(seed)
    
    cluster_items = list(clusters.items())
    random.shuffle(cluster_items)

    total_rows = sum(header_counts.values())
    train_target = total_rows * train_ratio
    valid_target = total_rows * (train_ratio + valid_ratio)

    seq_split: dict[str, str] = {}
    cumulative = 0

    for rep, members in cluster_items:
        cluster_row_count = sum(header_counts[member] for member in members)
        
        if cumulative < train_target:
            label = "train"
        elif cumulative < valid_target:
            label = "valid"
        else:
            label = "test"
            
        cumulative += cluster_row_count
        
        for member in members:
            seq_split[member] = label

    return seq_split


def main():
    print(f"Loading CSV: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    print(f"  Rows: {len(df)}, unique sequences: {df['sequence'].nunique()}")

    seq_counts = df["sequence"].value_counts().to_dict()

    unique_seqs = list(seq_counts.keys())
    seq_to_header = {seq: f"seq{i}" for i, seq in enumerate(unique_seqs)}
    
    header_counts = {seq_to_header[seq]: count for seq, count in seq_counts.items()}

    with tempfile.TemporaryDirectory(prefix="mmseqs_kcat_") as workdir:
        fasta_path = os.path.join(workdir, "sequences.fasta")
        write_fasta(unique_seqs, fasta_path)
        print(f"  Written FASTA with {len(unique_seqs)} sequences to {fasta_path}")

        for threshold in THRESHOLDS:
            pct = int(threshold * 100)
            col_name = f"{pct}_split"
            print(f"\n=== Clustering at {pct}% sequence identity ===")

            out_prefix = os.path.join(workdir, f"cluster_{pct}")
            tmp_dir = os.path.join(workdir, f"tmp_{pct}")
            os.makedirs(tmp_dir, exist_ok=True)

            tsv_path = run_mmseqs_cluster(fasta_path, out_prefix, tmp_dir, threshold)
            clusters = parse_clusters(tsv_path)

            print(f"  Found {len(clusters)} clusters")

            seq_split = assign_train_valid_test(clusters, header_counts, TRAIN_RATIO, VALID_RATIO, SEED)

            df[col_name] = df["sequence"].map(lambda s: seq_split[seq_to_header[s]])

            counts = df[col_name].value_counts()
            print(f"  Split distribution for {col_name}:")
            for split_label in ["train", "valid", "test"]:
                n = counts.get(split_label, 0)
                print(f"    {split_label}: {n} ({n/len(df)*100:.1f}%)")

    print(f"\nSaving to {OUTPUT_PATH}")
    df.to_csv(OUTPUT_PATH, index=False)
    print("Done.")


if __name__ == "__main__":
    main()