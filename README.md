# Enzyme Regime Benchmark

The repository contains two complementary benchmarks:

- **Global benchmark** (`global_benchmark.py`): Global enzyme turnover number (kcat) prediction across diverse enzymes, evaluating multiple ML models and protein/substrate embedding combinations.
- **Local benchmark** (`local_benchmark.ipynb`): Analysis of alpha-amylase specific activity prediction

---

## Repository Structure

```
enzyme-regime-benchmark/
├── global_benchmark.py         # Global enzyme kcat benchmark
├── local_benchmark.ipynb       # Alpha-amylase regime analysis
├── extract_features.py         # Embedding extraction (ProtT5, ESM2, MolFormer, ChemBERTa, ChemGPT)
├── save_model.py               # Train TabPFN and save the model weight
├── inference.py                # Load saved model and predict (single sample or CSV batch)
├── utils.py                    # Shared utilities (metrics, embedding loader)
├── data_preparation/
│   ├── brenda.py               # Scrape turnover number data from BRENDA
│   ├── preprocess_brenda.ipynb # Preprocess raw BRENDA data
│   ├── preprocess_sabiork.ipynb # Preprocess raw SABIO-RK data
│   ├── combine_brenda_sabiork.ipynb  # Merge BRENDA and SABIO-RK datasets
│   ├── add_seq_sub_split.py    # Add sequences, substrates, and split labels
│   ├── mmseqs_split.py         # Sequence-based train/test split using MMseqs2
│   └── split_dataset.ipynb     # Final dataset splitting
├── data/
│   ├── environmental_factors_split_info.csv   # Global benchmark dataset
│   └── alpha_amylase_combined.csv             # Alpha-amylase variant dataset
├── embeddings/                 # Download from Zenodo and place here (see below)
│   ├── prott5_features.npy                    # ProtT5 (global)
│   ├── esm2_15B_features.npy                  # ESM2-15B (global)
│   ├── esm2_3B_features.npy                   # ESM2-3B (global)
│   ├── ism_features.npy                       # ISM (global)
│   ├── molformer_features.npy                 # MolFormer (global)
│   ├── chemberta_features.npy                 # ChemBERTa (global)
│   ├── chemgpt_features.npy                   # ChemGPT (global)
│   ├── simson_features.npy                    # SimSon (global)
│   └── alpha_amylase_prott5_features.npy      # ProtT5 (local)
└── results/                    # Output CSVs from global benchmark runs
```

---

## Requirements

All experiments were run on an **NVIDIA H200 GPU** with **CUDA 12.6**.

Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Core packages:

- `torch==2.8.0+cu126`
- `tensorflow==2.21.0`
- `transformers==4.48.1`
- `scikit-learn==1.7.2`
- `xgboost==3.2.0`
- `tabpfn==6.4.1`
- `esm==3.2.1.post1`
- `scipy==1.15.3`
- `huggingface_hub==0.36.2`
- `joblib==1.5.3`
- `rdkit`
- `tqdm`
- `numpy`, `pandas`

> **Note:** `torch-geometric` and `x-transformers` are required only for the **SimSon** substrate embedding. SimSon embeddings must be extracted separately using the [SimSon repository](https://github.com/lee00206/SimSon) before running any experiment that uses the `--simson` flag.

> **Note:** **ISM** (in-silico mutagenesis) embeddings must also be extracted separately using the [ISM repository](https://github.com/jozhang97/ism) before running any experiment that uses the `--ism` flag.

> **Note:** `tabpfn==6.4.1` runs without an API token. Versions ≥ 8.x require a token from [ux.priorlabs.ai](https://ux.priorlabs.ai).

---

## Data & Model

Precomputed embeddings and the pretrained TabPFN model are available on Zenodo:

> **Zenodo**: [https://doi.org/10.5281/zenodo.20826133](https://doi.org/10.5281/zenodo.20826133)

The Zenodo record contains:
- All embedding files for `embeddings/` (ProtT5, ESM2-15B, ESM2-3B, ISM, MolFormer, ChemBERTa, ChemGPT, SimSon)
- A pretrained TabPFN model trained on the full global benchmark dataset

Download the files and place them under `embeddings/` before running any benchmark.

---

## Global Benchmark

Predicts enzyme turnover number (log₁₀ kcat) from protein and substrate embeddings, with optional environmental features (pH, temperature).

### Models

| Model | Key |
|---|---|
| Ridge Regression | `Ridge` |
| Random Forest | `RandomForest` |
| XGBoost | `XGBoost` |
| K-Nearest Neighbors | `KNN` |
| Extra Trees | `ExtraTrees` |
| TabPFN | `TabPFN` |

### Features

| Flag | Description |
|---|---|
| `--prott5` | ProtT5 protein embedding (1024-dim) |
| `--esm2_15B` | ESM2-15B protein embedding |
| `--esm2_3B` | ESM2-3B protein embedding |
| `--ism` | In-silico mutagenesis features |
| `--molformer` | MolFormer substrate embedding |
| `--chemberta` | ChemBERTa substrate embedding |
| `--chemgpt` | ChemGPT substrate embedding |
| `--simson` | SimSon substrate embedding |
| `--pH` | pH (raw + normalized) |
| `--temperature` | Temperature (raw, Kelvin, inverse, normalized) |

### Evaluation Modes

```bash
# All models, standard train/test split
python global_benchmark.py --prott5 --molformer --pH --temperature --mode all

# Single model
python global_benchmark.py --prott5 --molformer --pH --temperature --mode single --model TabPFN

# OOD evaluation (temperature / pH shift)
python global_benchmark.py --prott5 --molformer --pH --temperature --mode ood --model TabPFN

# 5-fold cross-validation per EC class
python global_benchmark.py --prott5 --molformer --pH --temperature --mode ec_class --model TabPFN

# Few-shot generalization across EC classes (n_shots ∈ {0, 10, 30, 100})
python global_benchmark.py --prott5 --molformer --pH --temperature --mode ec_few_shot --model TabPFN
```

### Additional Options

| Option | Default | Description |
|---|---|---|
| `--num_trials` | `3` | Number of random seeds |
| `--split_column` | `benchmark_split` | `benchmark_split` or `full_conditioned_split` |

Results are saved to `results/`.

---

## Local Benchmark
### Dataset

`data/alpha_amylase_combined.csv` is a preprocessed dataset containing single- and multi-mutation alpha-amylase variants (order 1–11) with measured specific activity. The raw data (`train.csv` and `test.csv`) is available from the [PET Pilot 2023 repository](https://github.com/the-protein-engineering-tournament/pet-pilot-2023/tree/main/in_silico_supervised/input/Alpha-Amylase%20(In%20Silico_%20Supervised)). Protein embeddings are precomputed in `embeddings/alpha_amylase_prott5_features.npy`.

### Experiments

| # | Title | Key Question |
|---|---|---|
| 1 | Common target | Does training on higher-order variants outperform lower-order training when predicting order-t variants? |
| 2 | Mutational distance and order | Does performance decay with mutational distance? Does mutation order matter after controlling for distance? |
| 3 | Single → multi-mutation generalization | Can a model trained on single mutants predict multi-mutation variants, given full single-mutation coverage? |
| 4 | Directional asymmetry | Is generalizing from low-order → high-order easier or harder than high-order → low-order? |
| 5 | Epistasis | Does a TabPFN model trained on single mutants outperform an additive (linear) baseline on multi-mutation variants? |
| 6 | Pair-covered vs. pair-uncovered | Does including training variants that share a mutation pair with the test variant improve prediction? |
| 7 | Exact substitution vs. position coverage | Does knowing the exact amino acid matter beyond knowing which positions are mutated? |
| 8 | Exact substitution identity | Does exact-token training proximity (same AA) outperform position-only proximity after controlling for positional distance? |
| 9 | Position-holdout | How does performance degrade as the number of completely unobserved mutation positions increases? |
| 10 | Pair-coverage within the same mutational distance | Does mutation-pair coverage in the training set explain prediction accuracy for multi-mutation variants, after controlling for mutational distance? |
| 11 | Prediction accuracy and mutational position coverage | How does prediction accuracy decay as a function of positional distance to the nearest training variant? |

---

## Training & Inference

### Extract Embeddings

Before training, precompute embeddings with `extract_features.py`:

```bash
python extract_features.py \
    --input_csv data/environmental_factors_split_info.csv \
    --protein_model_checkpoint Rostlab/prot_t5_xl_uniref50 \
    --substrate_model_checkpoint DeepChem/ChemBERTa-77M-MLM
```

> **Note:** `extract_features.py` contains a HuggingFace API token line near the top (`HfFolder.save_token(...)`). Replace the placeholder value with your own token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) before running.

Extracted embeddings are saved to `embeddings/`.

### Train and Save a Model

`save_model.py` trains TabPFN on the global benchmark dataset and saves the fitted model (including training data and feature configuration) to a `.joblib` file:

```bash
python save_model.py --prott5 --molformer --pH --temperature
# → models/tabpfn_pH_T_prott5_molformer_.joblib
```

| Option | Default | Description |
|---|---|---|
| `--seed` | `1024` | Random seed |
| `--save_path` | `models` | Directory to save the model |
| Feature flags | — | Same as Global Benchmark (`--prott5`, `--molformer`, etc.) |

> **Note:** TabPFN uses in-context learning, so the saved file contains the training data alongside the pretrained model weights. Feature flags are stored in the checkpoint so inference automatically uses the same pipeline.

### Inference

`inference.py` loads a saved model and predicts kcat (log₁₀ scale and raw s⁻¹).

**Single sample:**

```bash
python inference.py \
    --model_path models/tabpfn_pH_T_prott5_molformer.joblib \
    --sequence MKTAYIAKQR... \
    --smiles "C(C1C(C(C(C(O1)O)O)O)O)O" \
    --pH 7.0 --temperature 40.0
```

**Batch from CSV:**

```bash
# pH and temperature from CSV columns
python inference.py \
    --model_path models/tabpfn_pH_T_prott5_molformer.joblib \
    --input_csv data/my_enzymes.csv \
    --seq_col sequence --smiles_col smiles \
    --ph_col pH --temperature_col temperature \
    --output_csv data/my_enzymes_predicted.csv

# Fixed pH and temperature for all rows
python inference.py \
    --model_path models/tabpfn_pH_T_prott5_molformer.joblib \
    --input_csv data/my_enzymes.csv \
    --seq_col sequence --smiles_col smiles \
    --pH 7.0 --temperature 40.0
```

The output CSV is the input CSV with two columns appended: `log10_kcat_pred` and `kcat_pred`. If `--output_csv` is omitted, results are saved as `{input}_predicted.csv`.

| Option | Default | Description |
|---|---|---|
| `--model_path` | 'models/tabpfn_pH_T_prott5_molformer.joblib' | Path to `.joblib` from `save_model.py` |
| `--sequence` | — | Amino acid sequence (single-sample mode) |
| `--smiles` | — | Substrate SMILES (single-sample mode) |
| `--input_csv` | — | Input CSV file (batch mode) |
| `--seq_col` | `sequence` | Column name for sequences |
| `--smiles_col` | `smiles` | Column name for SMILES |
| `--ph_col` | `None` | Column name for pH (optional) |
| `--temperature_col` | `None` | Column name for temperature (optional) |
| `--pH` | `7.0` | Default pH when column not provided |
| `--temperature` | `37.0` | Default temperature (°C) when column not provided |
| `--output_csv` | auto | Output path (default: `{input}_predicted.csv`) |
| `--device` | `cuda` | `cuda` or `cpu` |

---
