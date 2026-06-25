import argparse
import os
import joblib
import numpy as np
import pandas as pd
from extract_features import (
    ExtractProtT5Features,
    ExtractESM2Features,
    ExtractChemBERTaFeatures,
    ExtractMolFormer,
    extractChemGPT,
)


def _build_ph_features(pH_list) -> np.ndarray:
    pH = np.asarray(pH_list, float).reshape(-1, 1)
    return np.concatenate([pH, pH / 14], axis=1)


def _build_temperature_features(temperature_list) -> np.ndarray:
    T = np.asarray(temperature_list, float)
    temp_k        = T + 273.15
    temp_safe     = np.where(T == 0, 0.1, T)
    inv_temp      = 1 / temp_safe
    temp_k_norm   = T / 100
    inv_temp_min  = 1 / (100 + 273.15)
    inv_temp_max  = 1 / (25 + 273.15)
    inv_temp_norm = (inv_temp - inv_temp_min) / (inv_temp_max - inv_temp_min)
    return np.stack([T, temp_k, inv_temp, temp_k_norm, inv_temp_norm], axis=1)


def build_features(
    sequences,
    smiles_list,
    pH_list=None,
    temperature_list=None,
    feature_flags: dict = None,
    device: str = 'cuda',
) -> np.ndarray:
    n     = len(sequences)
    flags = feature_flags or {}
    parts = []

    def _broadcast(val, n):
        if val is None:
            return None
        if np.isscalar(val):
            return [val] * n
        return list(val)

    pH_list = _broadcast(pH_list, n)
    temperature_list = _broadcast(temperature_list, n)

    if flags.get('pH'):
        if pH_list is None:
            raise ValueError("pH values required but not provided.")
        parts.append(_build_ph_features(pH_list))

    if flags.get('temperature'):
        if temperature_list is None:
            raise ValueError("Temperature values required but not provided.")
        parts.append(_build_temperature_features(temperature_list))

    if flags.get('prott5'):
        extractor = ExtractProtT5Features(device, 'Rostlab/prot_t5_xl_uniref50')
        parts.append(extractor.forward(sequences, batch_size=16))

    if flags.get('esm2_15B'):
        extractor = ExtractESM2Features(device, 'facebook/esm2_t48_15B_UR50D')
        parts.append(extractor.forward(sequences, batch_size=4))

    if flags.get('esm2_3B'):
        extractor = ExtractESM2Features(device, 'facebook/esm2_t36_3B_UR50D')
        parts.append(extractor.forward(sequences, batch_size=8))

    if flags.get('chemberta'):
        extractor = ExtractChemBERTaFeatures(device, 'DeepChem/ChemBERTa-77M-MLM')
        parts.append(extractor.forward(smiles_list, batch_size=32))

    if flags.get('molformer'):
        extractor = ExtractMolFormer(device)
        parts.append(extractor.forward(smiles_list, batch_size=32))

    if flags.get('chemgpt'):
        parts.append(extractChemGPT(smiles_list, batch_size=16))

    if not parts:
        raise ValueError("No features selected. Check feature_flags in the saved model.")

    return np.concatenate(parts, axis=1)


def predict(
    model_path: str,
    sequence: str,
    smiles: str,
    pH: float = 7.0,
    temperature: float = 37.0,
    device: str = 'cuda',
) -> dict:
    """Predict kcat for a single (sequence, SMILES) pair."""
    checkpoint    = joblib.load(model_path)
    model         = checkpoint['model']
    feature_flags = checkpoint['feature_flags']
    print(f"Loaded model: features={checkpoint['used_features']}")

    X        = build_features([sequence], [smiles], [pH], [temperature], feature_flags, device)
    log_kcat = float(model.predict(X)[0])
    kcat     = 10 ** log_kcat

    print(f"\nPredicted log10(kcat) : {log_kcat:.4f}")
    print(f"Predicted kcat        : {kcat:.4f} s⁻¹")
    return {'log10_kcat': log_kcat, 'kcat': kcat}


def predict_batch(
    model_path: str,
    sequences,
    smiles_list,
    pH_list=None,
    temperature_list=None,
    device: str = 'cuda',
) -> pd.DataFrame:
    """
    Predict kcat for a batch of samples.

    Returns a DataFrame with columns [log10_kcat, kcat].
    """
    checkpoint    = joblib.load(model_path)
    model         = checkpoint['model']
    feature_flags = checkpoint['feature_flags']
    print(f"Loaded model: features={checkpoint['used_features']} | samples={len(sequences)}")

    X = build_features(sequences, smiles_list, pH_list, temperature_list, feature_flags, device)
    log_kcats = model.predict(X)
    kcats = 10 ** log_kcats

    return pd.DataFrame({'log10_kcat': log_kcats, 'kcat': kcats})


def predict_from_csv(
    model_path: str,
    input_csv: str,
    seq_col: str,
    smiles_col: str,
    ph_col: str = None,
    temperature_col: str = None,
    default_pH: float = 7.0,
    default_temperature: float = 40.0,
    output_csv: str = None,
    device: str = 'cuda',
) -> pd.DataFrame:
    """
    Read a CSV, run batch prediction, and save results.

    pH / temperature are taken from columns if specified,
    otherwise the default scalar values are used for all rows.
    """
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} rows from {input_csv}")

    sequences = df[seq_col].tolist()
    smiles_list = df[smiles_col].tolist()
    pH_list = df[ph_col].tolist() if ph_col in df.columns else default_pH
    temperature_list = df[temperature_col].tolist() if temperature_col in df.columns else default_temperature

    preds = predict_batch(model_path, sequences, smiles_list,
                          pH_list, temperature_list, device)

    df['log10_kcat_pred'] = preds['log10_kcat'].values
    df['kcat_pred']       = preds['kcat'].values

    if output_csv is None:
        output_csv = f"{input_csv.split('.')[0]}_predicted.csv"

    df.to_csv(output_csv, index=False)
    print(f"Results saved → {output_csv}")
    return df


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='models/tabpfn_pH_T_prott5_molformer.joblib', help='Path to .joblib file from save_model.py')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])

    # single-sample mode
    parser.add_argument('--sequence', help='Amino acid sequence (single-letter)')
    parser.add_argument('--smiles', help='Substrate SMILES string')
    parser.add_argument('--pH', type=float, default=7.0)
    parser.add_argument('--temperature', type=float, default=40.0, help='Temperature in °C')

    # batch (CSV) mode
    parser.add_argument('--input_csv', help='Input CSV file')
    parser.add_argument('--seq_col', default='sequence', help='Column name for sequences')
    parser.add_argument('--smiles_col', default='smiles', help='Column name for SMILES')
    parser.add_argument('--ph_col', default=None, help='Column name for pH (optional)')
    parser.add_argument('--temperature_col', default=None, help='Column name for temperature (optional)')
    parser.add_argument('--output_csv', default=None, help='Output CSV path (default: input_predicted.csv)')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.input_csv:
        predict_from_csv(
            model_path = args.model_path,
            input_csv = args.input_csv,
            seq_col = args.seq_col,
            smiles_col = args.smiles_col,
            ph_col = args.ph_col,
            temperature_col = args.temperature_col,
            default_pH = args.pH,
            default_temperature = args.temperature,
            output_csv = args.output_csv,
            device = args.device,
        )
    elif args.sequence and args.smiles:
        predict(
            model_path = args.model_path,
            sequence = args.sequence,
            smiles = args.smiles,
            pH = args.pH,
            temperature = args.temperature,
            device = args.device,
        )
    else:
        print("Error: provide either --sequence/--smiles (single) or --input_csv (batch).")
