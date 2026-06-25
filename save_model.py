import argparse
import math
import os
import joblib
import numpy as np
import pandas as pd
from tabpfn import TabPFNRegressor
from utils import set_seed, load_embeddings, compute_metrics


def main(args):
    set_seed(args.seed)
    os.makedirs(args.save_path, exist_ok=True)

    # load data
    df = pd.read_csv('data/environmental_factors_split_info.csv')
    pH = df['pH']
    temperature = df['temperature']

    labels = df['turnover_number'].replace(0, 1e-10)
    labels = np.array([math.log10(float(v)) for v in labels])

    input_features, used_features = load_embeddings(args, pH, temperature)
    print(f"Feature shape: {input_features.shape}  ({used_features})")

    model = TabPFNRegressor(device='cuda', ignore_pretraining_limits=True, random_state=args.seed)
    model.fit(input_features, labels)

    # save model
    feature_flags = {k: getattr(args, k) for k in
                     ['pH', 'temperature', 'prott5', 'esm2_15B', 'esm2_3B',
                      'ism', 'chemberta', 'molformer', 'chemgpt', 'simson']}

    save_path = os.path.join(args.save_path, f'tabpfn_{used_features[:-1]}.joblib')

    joblib.dump({
        'model': model,
        'used_features': used_features,
        'feature_flags': feature_flags,
        'seed': args.seed,
    }, save_path)
    print(f"Model saved → {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # features selection
    parser.add_argument('--pH', action='store_true')
    parser.add_argument('--temperature', action='store_true')
    parser.add_argument('--prott5', action='store_true')
    parser.add_argument('--esm2_15B', action='store_true')
    parser.add_argument('--esm2_3B', action='store_true')
    parser.add_argument('--ism', action='store_true')
    parser.add_argument('--chemberta', action='store_true')
    parser.add_argument('--molformer', action='store_true')
    parser.add_argument('--chemgpt', action='store_true')
    parser.add_argument('--simson', action='store_true')

    parser.add_argument('--seed', type=int, default=1024)
    parser.add_argument('--save_path', type=str, default='models')

    args = parser.parse_args()

    main(args)
