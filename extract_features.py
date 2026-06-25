import argparse
import pandas as pd
import numpy as np
import re
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
from transformers import AutoModel, AutoConfig, AutoTokenizer, T5Tokenizer, T5EncoderModel
import torch.nn as nn
from huggingface_hub.hf_api import HfFolder
import torch
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem

HfFolder.save_token(API_KEY)


def extract_esmc_features(sequences, normalize=True):
    client = ESMC.from_pretrained("esmc_600m").to("cuda")
    embeddings = []
    for sequence in tqdm(sequences, desc="Extracting ESM-C features"):
        protein = ESMProtein(sequence=sequence)
        protein_tensor = client.encode(protein)
        logits_output = client.logits(protein_tensor, LogitsConfig(sequence=True, return_embeddings=True))
        embedding = logits_output.embeddings[0, 0, :]
        
        if normalize:
            embedding = embedding / embedding.norm()
        
        embeddings.append(embedding.detach().cpu().numpy())
    
    return np.array(embeddings)


class ExtractChemBERTaFeatures:
    def __init__(self, device, substrate_model_checkpoint):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(substrate_model_checkpoint)

        config = AutoConfig.from_pretrained(substrate_model_checkpoint, output_hidden_states=True, resume_download=True)
        self.model = AutoModel.from_pretrained(
            substrate_model_checkpoint,
            config=config,
            cache_dir='./cache',
            resume_download=True,
            use_safetensors=True
        ).to(device)
        self.model.eval()

    
    def forward(self, smiles_list, batch_size=32):
        all_embeddings = []
        for i in tqdm(range(0, len(smiles_list), batch_size)):
            batch_smiles = smiles_list[i:i+batch_size]

            # tokenize
            encoded = self.tokenizer(
                batch_smiles,
                padding='max_length',
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)

            with torch.no_grad():
                output = self.model(input_ids=input_ids, attention_mask=attention_mask)
                embedding = output.pooler_output  # [batch_size, hidden_dim=384]
                # L2 normalization
                embedding = embedding / embedding.norm(dim=1, keepdim=True)
            
            all_embeddings.append(embedding.cpu().numpy())

        return np.vstack(all_embeddings)


class ExtractProtT5Features:
    def __init__(self, device, protein_model_checkpoint):
        self.device = device
        self.tokenizer = T5Tokenizer.from_pretrained(protein_model_checkpoint)

        config = AutoConfig.from_pretrained(protein_model_checkpoint, output_hidden_states=True, resume_download=True)
        self.model = T5EncoderModel.from_pretrained(
            protein_model_checkpoint,
            config=config,
            cache_dir='./cache',
            resume_download=True,
            use_safetensors=True
        ).to(device)
        self.model.eval()

    
    def forward(self, sequence_list, batch_size=32, normalize=True):
        sequence_list = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequence_list]
        all_embeddings = []
        for i in tqdm(range(0, len(sequence_list), batch_size)):
            ids = self.tokenizer.batch_encode_plus(sequence_list[i:i+batch_size], add_special_tokens=True, max_length=1024, pad_to_max_length=True)
            input_ids = torch.tensor(ids['input_ids']).to(self.device)
            attention_mask = torch.tensor(ids['attention_mask']).to(self.device)    

            with torch.no_grad():
                embedding_repr = self.model(input_ids=input_ids, attention_mask=attention_mask) # last hidden state shape: [batch_size, 9, 1024]
            
            for idx, sequence in enumerate(sequence_list[i:i+batch_size]):
                seq_len = (attention_mask[idx] == 1).sum()
                embedding = embedding_repr.last_hidden_state[idx, :seq_len-1, :]
                all_embeddings.append(embedding.cpu().numpy())
            
            del input_ids, attention_mask, embedding_repr
            torch.cuda.empty_cache()
            
        embedding_normalized = np.zeros([len(all_embeddings), len(all_embeddings[0][0])], dtype=float)

        for i in range(len(all_embeddings)): # [num_sequences, seq_len->dynamic, hidden_dim]
            for k in range(len(all_embeddings[0][0])): # hidden_dim
                for j in range(len(all_embeddings[i])): # [seq_len, hidden_dim]
                    embedding_normalized[i][k] += all_embeddings[i][j][k]
                embedding_normalized[i][k] /= len(all_embeddings[i])
        return embedding_normalized


class ExtractESM2Features:
    """ESM2 model"""
    def __init__(self, device, protein_model_checkpoint):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(protein_model_checkpoint)

        config = AutoConfig.from_pretrained(protein_model_checkpoint, output_hidden_states=True, resume_download=True)
        self.model = AutoModel.from_pretrained(
            protein_model_checkpoint,
            config=config,
            cache_dir='./cache',
            resume_download=True,
            use_safetensors=True
        ).to(device)
        self.model.eval()
    
    def forward(self, sequence_list, batch_size=32, normalize=True):
        all_embeddings = []
        for i in tqdm(range(0, len(sequence_list), batch_size)):
            batch_sequences = sequence_list[i:i+batch_size]

            # tokenize
            encoded = self.tokenizer(
                batch_sequences,
                padding='max_length',
                truncation=True,
                max_length=1024,
                return_tensors='pt'
            )
            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)

            with torch.no_grad():
                output = self.model(input_ids=input_ids, attention_mask=attention_mask)
                embedding = output.last_hidden_state[:, 0, :]  # [batch_size, hidden_dim=1280]
                
                if normalize:
                    # L2 normalization
                    embedding = embedding / embedding.norm(dim=1, keepdim=True)
            
            all_embeddings.append(embedding.cpu().numpy())

        return np.vstack(all_embeddings)


def extract_smiles_fingerprint(smiles_list, radius=2, nbits=2048):
    fingerprints = []
    except_idx = []
    for idx, smiles in tqdm(enumerate(smiles_list), desc="Extracting SMILES fingerprints"):
        try:
            fingerprint = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), radius, nbits)
            fingerprints.append(fingerprint)
        except:
            print(f"Error extracting fingerprint for {smiles}")
            fingerprints.append(np.zeros(nbits))
            except_idx.append(idx)
    return np.array(fingerprints), except_idx


class ExtractMolFormer(nn.Module):
    def __init__(self, device, model_checkpoint="ibm-research/MoLFormer-XL-both-10pct"):
        super().__init__()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_checkpoint,
            trust_remote_code=True,
            cache_dir='./cache',
        ).to(device)
        self.model.eval()

    def forward(self, smiles_list, batch_size=32, normalize=True):
        all_embeddings = []
        for i in tqdm(range(0, len(smiles_list), batch_size), desc="Extracting MolFormer features"):
            batch_smiles = smiles_list[i:i+batch_size]

            # tokenize
            encoded = self.tokenizer(
                batch_smiles,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)

            with torch.no_grad():
                output = self.model(input_ids=input_ids, attention_mask=attention_mask)
                # Mean pooling over sequence length (attention mask 고려)
                last_hidden = output.last_hidden_state  # [batch_size, seq_len, hidden_dim]
                mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
                sum_embeddings = torch.sum(last_hidden * mask_expanded, dim=1)
                sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
                embedding = sum_embeddings / sum_mask  # [batch_size, hidden_dim]

                if normalize:
                    # L2 normalization
                    embedding = embedding / embedding.norm(dim=1, keepdim=True)

            all_embeddings.append(embedding.cpu().numpy())

            del input_ids, attention_mask, output
            torch.cuda.empty_cache()

        return np.vstack(all_embeddings)


def extractChemGPT(smiles_list, batch_size=32, model_name='ncfrey/ChemGPT-1.2B'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model = AutoModel.from_pretrained(model_name, cache_dir='./cache').to(device)
    model.resize_token_embeddings(len(tokenizer))
    model.eval()

    all_embeddings = []
    for i in tqdm(range(0, len(smiles_list), batch_size), desc="Extracting ChemGPT features"):
        batch_smiles = smiles_list[i:i+batch_size]
        # tokenize
        encoded = tokenizer(
            batch_smiles,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt')
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            embedding = output.last_hidden_state[:, 0, :]  # [batch_size, hidden_dim=1536]
            # L2 normalization
            embedding = embedding / embedding.norm(dim=1, keepdim=True)
            all_embeddings.append(embedding.cpu().numpy())

    return np.vstack(all_embeddings)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--protein_model_checkpoint', type=str, default='Rostlab/prot_t5_xl_uniref50', help='Protein model checkpoint')
    parser.add_argument('--substrate_model_checkpoint', type=str, default='DeepChem/ChemBERTa-77M-MLM', help='Substrate model checkpoint')
    parser.add_argument('--input_csv', type=str, default='data/combined_brenda_sabiork_geometric_mean.csv', help='Input CSV file with sequences and SMILES')
    parser.add_argument('--output_protein_features', type=str, default='protein_features.npy', help='Output file for protein features')
    parser.add_argument('--output_smiles_features', type=str, default='smiles_features.npy', help='Output file for SMILES features')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = pd.read_csv(args.input_csv)
    sequences = data['sequences']
    substrates = data['canonical_smiles']


    # Extract protein features
    prot_extractor = ExtractProtT5Features(device, args.protein_model_checkpoint)
    protein_features = prot_extractor.forward(sequences, batch_size=16)
    np.save('embeddings/prott5_features.npy', protein_features)

    # Extract SMILES features
    smiles_extractor = ExtractChemBERTaFeatures(device, args.substrate_model_checkpoint)
    smiles_features = smiles_extractor.forward(smiles_list, batch_size=32)
    np.save('embeddings/chemberta_features.npy', smiles_features)

    # Extract ESM2 features
    args.protein_model_checkpoint = 'facebook/esm2_t36_3B_UR50D'
    esm2_extractor = ExtractESM2Features(device, args.protein_model_checkpoint)
    esm2_features = esm2_extractor.forward(sequences, batch_size=16)
    np.save('embeddings/esm2_3B_features.npy', esm2_features)

    args.protein_model_checkpoint = 'facebook/esm2_t48_15B_UR50D'
    esm2_extractor = ExtractESM2Features(device, args.protein_model_checkpoint)
    esm2_features = esm2_extractor.forward(sequences, batch_size=16)
    np.save('embeddings/esm2_15B_features.npy', esm2_features)

    # Extract MolFormer features
    molformer_extractor = ExtractMolFormer(device)
    molformer_features = molformer_extractor.forward(smiles_list, batch_size=32)
    np.save('embeddings/molformer_features.npy', molformer_features)

    # Extract ChemGPT features
    chemgpt_features = extractChemGPT(smiles_list, batch_size=16, model_name='ncfrey/ChemGPT-1.2B')
    np.save('embeddings/chemgpt_features.npy', chemgpt_features)

