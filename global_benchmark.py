# Global benchmark code
import pandas as pd
import numpy as np
import argparse
import os
import math
from tqdm import tqdm
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from tabpfn import TabPFNRegressor
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras import callbacks
from keras import optimizers
from keras.layers import BatchNormalization
from utils import *
from huggingface_hub.hf_api import HfFolder

HfFolder.save_token(API_KEY)


def model_list(seed):
    models = {
        "Ridge": Ridge(alpha=1.0, random_state=seed),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=seed),
        "XGBoost": XGBRegressor(n_estimators=100, random_state=seed),
        "KNN": KNeighborsRegressor(n_neighbors=5),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=100, random_state=seed),
        "TabPFN": TabPFNRegressor(device='cuda', ignore_pretraining_limits=True, random_state=seed)
    }
    return models


def mlp(train_features, train_labels, valid_features, valid_labels, test_features, test_labels):
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    valid_features = scaler.transform(valid_features)
    test_features  = scaler.transform(test_features)

    # build a model
    model = Sequential()
    model.add(Dense(64, input_dim=train_features.shape[1], activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(100, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(1, activation='linear'))

    earlystopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    optimizer = optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
    _ = model.fit(train_features, train_labels, epochs=100, batch_size=32, validation_data=(valid_features, valid_labels), callbacks=[earlystopping])
    predictions = model.predict(test_features)
    r2, rmse, _ = compute_metrics(test_labels, predictions)
    print(f"R2: {r2:.4f}, RMSE: {rmse:.4f}")

    return r2, rmse


def _get_ec_class(x):
    s = str(x).split('.')[0]
    return int(s) if s.isdigit() else None


def run_ood_mode(args, input_features, labels, df, split_column, used_features):
    os.makedirs('results', exist_ok=True)

    conditions = [
        ('temperature', (df['temperature'] > 22) & (df['temperature'] <= 60)),
        ('pH',          (df['pH'] > 6)           & (df['pH'] <= 9)),
    ]

    all_results = []
    for ood_type, id_mask in conditions:
        print(f"\n{'='*60}")
        print(f"OOD Experiment: {ood_type}  (model={args.model})")
        print(f"{'='*60}")

        train_mask = df[split_column] == 'train'
        test_mask = df[split_column] == 'test'

        train_idx = df[train_mask & id_mask].index.tolist()
        test_idx = df[test_mask].index.tolist()
        test_id_idx = df[test_mask & id_mask].index.tolist()
        test_ood_idx = df[test_mask & ~id_mask].index.tolist()

        print(f"Train(ID)={len(train_idx)}  Test(all)={len(test_idx)}  "
              f"Test-ID={len(test_id_idx)}  Test-OOD={len(test_ood_idx)}")

        if not train_idx:
            print("No ID training samples. Skipping.")
            continue

        X_train = input_features[train_idx]
        y_train = labels[train_idx]
        X_test  = input_features[test_idx]
        y_test  = labels[test_idx]

        pos_map = {idx: pos for pos, idx in enumerate(test_idx)}
        id_positions = [pos_map[i] for i in test_id_idx]
        ood_positions = [pos_map[i] for i in test_ood_idx]

        set_seed(1024)
        seeds = np.random.choice(10000, size=args.num_trials, replace=False).tolist()
        print(f"Seeds: {seeds}")

        rows = []
        for seed in seeds:
            set_seed(seed)
            model = model_list(seed)[args.model]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            r2_all, rmse_all, sp_all = compute_metrics(y_test, y_pred)
            row = {'ood_type': ood_type, 'seed': seed,
                   'all_R2': r2_all, 'all_RMSE': rmse_all, 'all_SpearmanR': sp_all}

            if len(id_positions) > 1:
                r2_id, rmse_id, sp_id = compute_metrics(y_test[id_positions], y_pred[id_positions])
                row.update({'id_R2': r2_id, 'id_RMSE': rmse_id, 'id_SpearmanR': sp_id})
            else:
                row.update({'id_R2': None, 'id_RMSE': None, 'id_SpearmanR': None})

            if len(ood_positions) > 1:
                r2_ood, rmse_ood, sp_ood = compute_metrics(y_test[ood_positions], y_pred[ood_positions])
                row.update({'ood_R2': r2_ood, 'ood_RMSE': rmse_ood, 'ood_SpearmanR': sp_ood})
            else:
                row.update({'ood_R2': None, 'ood_RMSE': None, 'ood_SpearmanR': None})

            id_str  = (f"R2={row.get('id_R2', 'N/A'):.4f} RMSE={row.get('id_RMSE', 'N/A'):.4f}"
                       if row['id_R2'] is not None else "N/A")
            ood_str = (f"R2={row.get('ood_R2', 'N/A'):.4f} RMSE={row.get('ood_RMSE', 'N/A'):.4f}"
                       if row['ood_R2'] is not None else "N/A")
            print(f"  seed={seed} | All: R2={r2_all:.4f} RMSE={rmse_all:.4f} | ID: {id_str} | OOD: {ood_str}")
            rows.append(row)

        result_df = pd.DataFrame(rows)
        print(f"\nSummary ({ood_type}):")
        for grp, label in [('all', 'All test'), ('id', 'ID test'), ('ood', 'OOD test')]:
            r2_col, rmse_col = f'{grp}_R2', f'{grp}_RMSE'
            valid = result_df[r2_col].dropna()
            if not valid.empty:
                print(f"  {label:10s}: R2={valid.mean():.4f}±{valid.std():.4f}  "
                      f"RMSE={result_df[rmse_col].dropna().mean():.4f}±{result_df[rmse_col].dropna().std():.4f}")

        out = f'results/ood_{ood_type}_{args.model}_{used_features}{split_column}.csv'
        result_df.to_csv(out, index=False)
        print(f"Results saved to {out}")
        all_results.append(result_df)

    return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()


def run_ec_class_mode(args, input_features, labels, df, used_features):
    df = df.copy()
    df['_ec_class'] = df['ec_number'].apply(_get_ec_class)

    print(f"\n{'='*60}")
    print(f"EC Class Mode — 5-fold CV per class  (model={args.model})")
    print(f"{'='*60}")

    rows = []
    for ec_class in range(1, 8):
        class_idx = df[df['_ec_class'] == ec_class].index.tolist()
        if len(class_idx) < 10:
            print(f"EC {ec_class}: {len(class_idx)} samples — skipping (< 10).")
            continue

        X_cls = input_features[class_idx]
        y_cls = labels[class_idx]

        kf = KFold(n_splits=5, shuffle=True, random_state=1024)
        fold_r2, fold_rmse, fold_sp = [], [], []

        for fold_i, (tr_pos, te_pos) in enumerate(kf.split(class_idx)):
            set_seed(1024)
            model = model_list(1024)[args.model]
            model.fit(X_cls[tr_pos], y_cls[tr_pos])
            y_pred = model.predict(X_cls[te_pos])
            r2, rmse, sp = compute_metrics(y_cls[te_pos], y_pred)
            fold_r2.append(r2); fold_rmse.append(rmse); fold_sp.append(sp)
            print(f"EC {ec_class} fold {fold_i+1}/5: R2={r2:.4f} RMSE={rmse:.4f} SpearmanR={sp:.4f}")

        print(f"EC {ec_class} ({len(class_idx)} samples) MEAN: "
              f"R2={np.mean(fold_r2):.4f}±{np.std(fold_r2):.4f}  "
              f"RMSE={np.mean(fold_rmse):.4f}±{np.std(fold_rmse):.4f}  "
              f"SpearmanR={np.mean(fold_sp):.4f}±{np.std(fold_sp):.4f}")
        rows.append({
            'EC_class': ec_class, 'n_samples': len(class_idx),
            'R2_mean': np.mean(fold_r2),   'R2_std': np.std(fold_r2),
            'RMSE_mean': np.mean(fold_rmse), 'RMSE_std': np.std(fold_rmse),
            'SpearmanR_mean': np.mean(fold_sp), 'SpearmanR_std': np.std(fold_sp),
        })

    results_df = pd.DataFrame(rows)
    print(f"\n{'─'*60}\n{results_df.to_string(index=False)}")

    out = f'results/ec_class_{args.model}_{used_features}.csv'
    results_df.to_csv(out, index=False)
    print(f"Results saved to {out}")
    return results_df


def run_ec_few_shot_mode(args, input_features, labels, df, used_features):
    df = df.copy()
    df['_ec_class'] = df['ec_number'].apply(_get_ec_class)
    valid_mask = df['_ec_class'].notna()
    ec_classes = sorted(int(c) for c in df.loc[valid_mask, '_ec_class'].unique())

    print(f"\n{'='*60}")
    print(f"EC Few-Shot Mode ({args.model}) — classes: {ec_classes}")
    print(f"{'='*60}")

    set_seed(1024)
    seeds = np.random.choice(10000, size=args.num_trials, replace=False).tolist()

    all_rows = []
    for n_shots in [0, 10, 30, 100]:
        print(f"\n--- n_shots = {n_shots} ---")
        for ec_class in ec_classes:
            class_idx = df[df['_ec_class'] == ec_class].index.tolist()
            if len(class_idx) <= n_shots:
                print(f"  EC {ec_class}: {len(class_idx)} samples ≤ {n_shots}, skipping.")
                continue

            other_idx = df[(df['_ec_class'] != ec_class) & valid_mask].index.tolist()

            rng = np.random.default_rng(1024)
            few_shot_idx = rng.choice(class_idx, n_shots, replace=False).tolist()
            few_shot_set = set(few_shot_idx)
            train_idx = other_idx + few_shot_idx
            test_idx  = [i for i in class_idx if i not in few_shot_set]

            X_train = input_features[train_idx]
            y_train = labels[train_idx]
            X_test  = input_features[test_idx]
            y_test  = labels[test_idx]

            r2_list, rmse_list, sp_list = [], [], []
            for seed in seeds:
                set_seed(seed)
                model = model_list(seed)[args.model]
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                r2, rmse, sp = compute_metrics(y_test, y_pred)
                r2_list.append(r2); rmse_list.append(rmse); sp_list.append(sp)

            print(f"EC {ec_class} | n_shots={n_shots} | test_n={len(test_idx)} | "
                  f"R2={np.mean(r2_list):.4f}±{np.std(r2_list):.4f}  "
                  f"RMSE={np.mean(rmse_list):.4f}±{np.std(rmse_list):.4f}  "
                  f"SpearmanR={np.mean(sp_list):.4f}±{np.std(sp_list):.4f}")
            all_rows.append({
                'n_shots': n_shots, 'EC_class': ec_class,
                'n_train_total': len(train_idx), 'n_test': len(test_idx),
                'R2_mean': np.mean(r2_list),   'R2_std': np.std(r2_list),
                'RMSE_mean': np.mean(rmse_list), 'RMSE_std': np.std(rmse_list),
                'SpearmanR_mean': np.mean(sp_list), 'SpearmanR_std': np.std(sp_list),
            })

    results_df = pd.DataFrame(all_rows)
    print(f"\n{'─'*60}\n{results_df.to_string(index=False)}")

    out = f'results/ec_few_shot_{args.model}_{used_features}.csv'
    results_df.to_csv(out, index=False)
    print(f"Results saved to {out}")
    return results_df


def train_models(X_train, y_train, X_test, y_test, num_trials=5):
    set_seed(1024)
    seeds = np.random.choice(10000, size=num_trials, replace=False).tolist()
    print("Selected seeds: ", seeds)

    model_names = list(model_list(seeds[0]).keys())
    results_dict = {model_name: {'r2': [], 'rmse': []} for model_name in model_names}

    print("Train length: ", len(X_train))
    print("Test length: ", len(X_test))

    for seed in seeds:
        print(f"\n{'='*50}")
        print(f"Training with seed: {seed}")
        print(f"{'='*50}")
        set_seed(seed)
        models = model_list(seed)
        for model_name, model in tqdm(models.items(), desc=f'Training models with seed {seed}'):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            r2, rmse, _ = compute_metrics(y_test, y_pred)

            results_dict[model_name]['r2'].append(r2)
            results_dict[model_name]['rmse'].append(rmse)

            print(f'{model_name}: R2={r2:.4f}, RMSE={rmse:.4f}')

    columns = pd.MultiIndex.from_product([model_names, ['r2', 'rmse']])
    df_data = []
    for i, seed in enumerate(seeds):
        row = []
        for model_name in model_names:
            row.extend([results_dict[model_name]['r2'][i], results_dict[model_name]['rmse'][i]])
        df_data.append(row)

    results_df = pd.DataFrame(df_data, index=seeds, columns=columns)
    results_df.index.name = 'seed'

    mean_row = []
    std_row = []
    for model_name in model_names:
        mean_row.extend([np.mean(results_dict[model_name]['r2']), np.mean(results_dict[model_name]['rmse'])])
        std_row.extend([np.std(results_dict[model_name]['r2']), np.std(results_dict[model_name]['rmse'])])

    results_df.loc['mean'] = mean_row
    results_df.loc['std'] = std_row

    return results_df


def train_single_model(args, X_train, y_train, X_test, y_test, num_trials=3):
    set_seed(1024)
    seeds = np.random.choice(10000, size=num_trials, replace=False).tolist()
    print("Selected seeds: ", seeds)

    r2_scores = []
    rmse_scores = []

    indices = seeds + ['mean', 'std']
    results_df = pd.DataFrame(index=indices, columns=['R2', 'RMSE'])
    for seed in tqdm(seeds, desc=f'Training single model {args.model}'):
        print(f"\n{'='*50}")
        print(f"Training with seed: {seed}")
        print(f"{'='*50}")
        set_seed(seed)

        model = model_list(seed)[args.model]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2, rmse, _ = compute_metrics(y_test, y_pred)

        r2_scores.append(r2)
        rmse_scores.append(rmse)

        print(f"{seed} - {args.model}: R2={r2:.4f}, RMSE={rmse:.4f}")

    r2_scores.append(np.mean(r2_scores))
    r2_scores.append(np.std(r2_scores))
    rmse_scores.append(np.mean(rmse_scores))
    rmse_scores.append(np.std(rmse_scores))
    results_df['R2'] = r2_scores
    results_df['RMSE'] = rmse_scores

    return results_df    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # feature selection
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

    # run parameters
    parser.add_argument('--num_trials', default=3, type=int)
    parser.add_argument('--mode', default='all',
                        choices=['all', 'single', 'ood', 'ec_class', 'ec_few_shot'])
    parser.add_argument('--model', default='TabPFN',
                        choices=['Ridge', 'RandomForest', 'XGBoost', 'KNN', 'ExtraTrees', 'TabPFN'])
    parser.add_argument('--split_column', default='benchmark_split',
                        choices=['benchmark_split', 'full_conditioned_split'])

    args = parser.parse_args()

    # results directory
    os.makedirs('results', exist_ok=True)

    # load data
    df = pd.read_csv('data/environmental_factors_split_info.csv')

    pH, temperature = df.pH, df.temperature

    labels = df.turnover_number.replace(0, 1e-10)
    labels = [math.log10(float(label)) for label in labels]
    labels = np.array(labels)

    # load features
    input_features, used_features = load_embeddings(args, pH, temperature)

    # split data
    split_column = args.split_column
    train_idx = df[df[split_column] == 'train'].index.tolist()
    valid_idx = df[df[split_column] == 'valid'].index.tolist()
    test_idx = df[df[split_column] == 'test'].index.tolist()

    X_valid, y_valid = input_features[valid_idx], labels[valid_idx]

    X_train, y_train = input_features[train_idx], labels[train_idx]
    X_test, y_test = input_features[test_idx], labels[test_idx]

    if args.mode == 'all':
        results_df = train_models(X_train, y_train, X_test, y_test, num_trials=args.num_trials)
        results_df.to_csv(f'results/{used_features}results_{split_column}.csv')
    elif args.mode == 'single':
        results_df = train_single_model(args, X_train, y_train, X_test, y_test, num_trials=args.num_trials)
        results_df.to_csv(f'results/{used_features}results_{args.model}_{split_column}.csv')
    elif args.mode == 'ood':
        run_ood_mode(args, input_features, labels, df, split_column, used_features)
    elif args.mode == 'ec_class':
        run_ec_class_mode(args, input_features, labels, df, used_features)
    elif args.mode == 'ec_few_shot':
        run_ec_few_shot_mode(args, input_features, labels, df, used_features)

