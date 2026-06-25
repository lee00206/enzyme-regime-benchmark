import os
import numpy as np
import pandas as pd
import random
import torch
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error



def set_seed(seed):
    np.seterr(all="ignore")
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_embeddings(args, pH, temperature):
    # concatenate selected features
    used_features = ''

    input_features = []
    if args.pH:
        pH_norm = pH / 14  # normalized pH (0-1 range)

        input_features.append(np.array(pH).reshape(-1, 1))
        input_features.append(np.array(pH_norm).reshape(-1, 1))
        used_features += 'pH_'

    if args.temperature:
        temp_k = temperature + 273.15
        temp_safe = temperature.replace(0, 0.1)
        inv_temp = 1/temp_safe
        temp_k_norm = temperature / 100
        inv_temp_min = 1 / (100 + 273.15)
        inv_temp_max = 1 / (25 + 273.15)
        inv_temp_norm = (inv_temp - inv_temp_min) / (inv_temp_max - inv_temp_min)
        input_features.append(np.array(temperature).reshape(-1, 1))
        input_features.append(np.array(temp_k).reshape(-1,1))
        input_features.append(np.array(inv_temp).reshape(-1,1))
        input_features.append(np.array(temp_k_norm).reshape(-1,1))
        input_features.append(np.array(inv_temp_norm).reshape(-1,1))
        used_features += 'T_'

    if args.prott5:
        prott5_features = np.load('embeddings/prott5_features.npy')
        print('ProtT5: ', prott5_features.shape)
        input_features.append(prott5_features)
        used_features += 'prott5_'

    if args.esm2_15B:
        esm2_15B_features = np.load('embeddings/esm2_15B_features.npy')
        print('ESM2 15B: ', esm2_15B_features.shape)
        input_features.append(esm2_15B_features)
        used_features += 'esm2_15B_'

    if args.esm2_3B:
        esm2_3B_features = np.load('embeddings/esm2_3B_features.npy')
        print('ESM2 3B: ', esm2_3B_features.shape)
        input_features.append(esm2_3B_features)
        used_features += 'esm2_3B_'

    if args.ism:
        ism_features = np.load('embeddings/ism_features.npy')
        print('ISM: ', ism_features.shape)
        input_features.append(ism_features)
        used_features += 'ism_'

    if args.chemberta:
        chemberta_features = np.load('embeddings/chemberta_features.npy')
        print('ChemBERTa: ', chemberta_features.shape)
        input_features.append(chemberta_features)
        used_features += 'chemberta_'

    if args.molformer:
        molformer_features = np.load('embeddings/molformer_features.npy')
        print('MolFormer: ', molformer_features.shape)
        input_features.append(molformer_features)
        used_features += 'molformer_'

    if args.chemgpt:
        chemgpt_features = np.load('embeddings/chemgpt_features.npy')
        print('ChemGPT: ', chemgpt_features.shape)
        input_features.append(chemgpt_features)
        used_features += 'chemgpt_'

    if args.simson:
        simson_features = np.load('embeddings/simson_features.npy')
        print('SimSon: ', simson_features.shape)
        input_features.append(simson_features)
        used_features += 'simson_'

    input_features = np.concatenate(input_features, axis=1)
    print("Input features shape:", input_features.shape)

    return input_features, used_features


def compute_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    spearman_r, _ = stats.spearmanr(y_true, y_pred)
    return r2, rmse, spearman_r









