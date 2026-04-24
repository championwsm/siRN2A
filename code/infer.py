import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np
import pandas as pd
import scipy
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from collections import Counter
from sklearn.metrics import precision_score, recall_score, mean_absolute_error, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, Normalizer

import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
from Bio.Seq import Seq
from tqdm import tqdm
import resource
import itertools
import json

import argparse

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters()
                        if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


class GenomicTokenizer:
    def __init__(self, ngram=5, stride=2):
        self.ngram = ngram
        self.stride = stride

    def tokenize(self, t):
        t = t.upper()
        if self.ngram == 1:
            toks = list(t)
        else:
            toks = [
                t[i:i + self.ngram] for i in range(0, len(t), self.stride)
                if len(t[i:i + self.ngram]) == self.ngram
            ]
        if len(toks[-1]) < self.ngram:
            toks = toks[:-1]
        return toks


class GenomicVocab:
    def __init__(self, itos):
        self.itos = itos
        self.stoi = {v: k for k, v in enumerate(self.itos)}

    @classmethod
    def create(cls, tokens, max_vocab, min_freq):
        freq = Counter(tokens)
        itos = ['<pad>'] + [
            o for o, c in freq.most_common(max_vocab - 1) if c >= min_freq
        ]
        return cls(itos)


class SiRNADataset2(Dataset):
    def __init__(self, df, columns_siRNA, vocab, tokenizer, max_len_siRNA,
                 columns_mRNA, mRNA_embedding_orthrus, thermodynamics_embedding):
        self.columns_siRNA = columns_siRNA
        self.max_len_siRNA = max_len_siRNA
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.columns_mRNA = columns_mRNA
        self.mRNA_embedding_orthrus = mRNA_embedding_orthrus

        self.data = df.to_dict(orient='records')
        self.encoded_siRNAs = [
            [self.tokenize_and_encode_siRNA(row[col]) for col in self.columns_siRNA]
            for row in self.data
        ]
        self.mRNA_tensors = [
            [torch.tensor(row[col]) for col in self.mRNA_embedding_orthrus]
            for row in self.data
        ]

        self.precomputed_features = [
            (
                torch.tensor(row['siRNA_concentration'], dtype=torch.float),
                torch.tensor(row['cell_line_donor'], dtype=torch.long),
                torch.tensor(row['Transfection_method'], dtype=torch.long),
                torch.tensor(row['Duration_after_transfection_h'], dtype=torch.float),
            )
            for row in self.data
        ]
        self.targets = [torch.tensor(row['mRNA_remaining_pct'], dtype=torch.float)
                        for row in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seqs = self.encoded_siRNAs[idx] + self.mRNA_tensors[idx]
        seqs.extend(self.precomputed_features[idx])
        target = self.targets[idx]
        return seqs, target

    def tokenize_and_encode_siRNA(self, seq):
        if ' ' in seq:
            tokens = seq.split()
        else:
            tokens = self.tokenizer.tokenize(seq)
        encoded = [self.vocab.stoi.get(token, 0) for token in tokens]
        padded = encoded + [0] * (self.max_len_siRNA - len(encoded))
        return torch.tensor(padded[:self.max_len_siRNA], dtype=torch.long)


class SiRNAEncoderWithAblation(nn.Module):
    def __init__(self, vocab_size, embed_dim, cell_line_size, method_size, duration_size,
                 feature_mask):
        super(SiRNAEncoderWithAblation, self).__init__()
        self.embed_dim = embed_dim

        self.feature_mask = feature_mask

        # Embedding & Linear Layers
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.cell_line_embedding = nn.Embedding(cell_line_size, embed_dim, padding_idx=0)
        self.transfection_method_embedding = nn.Embedding(method_size, embed_dim, padding_idx=0)
        self.duration_embedding = nn.Linear(1, embed_dim)
        self.fc_concentration = nn.Linear(1, embed_dim)

        self.gru = nn.GRU(embed_dim, embed_dim, num_layers=3, bidirectional=True, batch_first=True, dropout=0.)

        self.fc_mlp = nn.Linear(4 * embed_dim, 4 * embed_dim)
        self.fc_gru = nn.Linear(8 * embed_dim, 4 * embed_dim)       # seq count

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.)

    def maybe_append(self, features, tensor, name):
        if self.feature_mask[name]:
            features.append(tensor)
        else:
            features.append(torch.zeros(tensor.shape[0], self.embed_dim, device=tensor.device))
            # print('setting', name, 'to 0')

    def forward(self, x):
        embedded_siRNA = [self.embedding(seq) for seq in x[:-260]]
        embedded_mRNA_raw = [seq.reshape(-1, 1).float() for seq in x[-260:-4]]
        embedded_mRNA = torch.cat(embedded_mRNA_raw, dim=1)

        # --- Ablation Features ---
        features = []
        self.maybe_append(features, self.fc_concentration(x[-4].reshape(-1, 1)), 'concentration')
        self.maybe_append(features, self.cell_line_embedding(x[-3]), 'cell_line')
        self.maybe_append(features, self.transfection_method_embedding(x[-2]), 'transfection_method')
        self.maybe_append(features, self.duration_embedding(x[-1].reshape(-1, 1)),'duration')

        if not self.feature_mask['mRNA']:
            embedded_mRNA = torch.zeros(embedded_mRNA.shape[0], self.embed_dim, device=embedded_mRNA.device)

        mlp_emb = torch.cat(features, dim=1)  # shape = (B, 10 * embed_dim)

        # --- GRU Sequence Part ---
        outputs = []
        for embed in embedded_siRNA:
            x_gru, _ = self.gru(embed)
            x_gru = self.dropout(torch.mean(x_gru, axis=1))  # mean pooling
            outputs.append(x_gru)
        embedded_siRNA = torch.cat(outputs, dim=1)  # shape = (B, 8 * embed_dim)

        # --- Combine ---
        mlp_emb = self.relu(self.fc_mlp(mlp_emb))
        embedded_siRNA = self.relu(self.fc_gru(embedded_siRNA))
        all_emb = torch.cat([mlp_emb, embedded_siRNA, embedded_mRNA], dim=1)

        return all_emb


class ContrastiveEncoder(nn.Module):

    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim * 8 + 256, hidden_dim * 4)
        self.fc2 = nn.Linear(4 * hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.ln1 = nn.LayerNorm(hidden_dim * 4)

    def forward(self, x, additional=None):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.ln1(x)
        x = self.fc2(x)
        return x

class Decoder(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, embed_dim // 2)
        self.fc2 = nn.Linear(embed_dim // 2, 1)
        self.relu = nn.ReLU()
        self.ln1 = nn.LayerNorm(embed_dim // 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.ln1(x)
        x = self.fc2(x)
        return x


class Predictor(nn.Module):

    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, 1)

    def forward(self, x, additional=None):
        x = self.fc1(x)
        x = 100 * torch.sigmoid(x)
        return x


class SiRNAModel(nn.Module):

    def __init__(self,
                 vocab_size,
                 embed_dim,
                 cell_line_size,
                 method_size,
                 duration_size,
                 feature_mask,
                 hidden_dim=256):
        super(SiRNAModel, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.transform = SiRNAEncoderWithAblation(vocab_size, embed_dim,
                                                  cell_line_size, method_size,
                                                  duration_size, feature_mask)
        self.encoder = ContrastiveEncoder(embed_dim, hidden_dim)
        self.decoder = Decoder(3 * self.embed_dim, self.hidden_dim)
        self.predictor = Predictor(self.embed_dim, self.hidden_dim)

    def forward(self, x):
        encoded_features = self.transform(x)
        x = self.encoder(encoded_features)
        return x


def calculate_metrics(y_true, y_pred, threshold=30, top_k=5):
    if isinstance(y_true, pd.DataFrame) or isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.DataFrame) or isinstance(y_pred, pd.Series):
        y_pred = y_pred.values
    if y_true.ndim > 1:
        y_true = y_true.ravel()
    if y_pred.ndim > 1:
        y_pred = y_pred.ravel()
    mae = mean_absolute_error(y_true, y_pred)

    if y_true.shape[0] < 2 or y_pred.shape[0] < 2:
        pcc = np.nan
        spcc = np.nan
    else:
        pcc = scipy.stats.pearsonr(y_true, y_pred).statistic
        spcc = scipy.stats.spearmanr(y_true, y_pred).correlation

    postive_idx = torch.where(torch.tensor(y_true) <= threshold)[0]

    relevant_idx = torch.argsort(torch.tensor(y_true),
                                 dim=-1,
                                 descending=False)[:top_k]
    target = torch.zeros_like(torch.tensor(y_true))
    target[relevant_idx] = 1.
    preds = torch.tensor(y_pred)
    precision_topk = target[preds.topk(min(top_k, preds.shape[-1]),
                                       dim=-1,
                                       largest=False)[1]].sum().float() / top_k
    precision_topk = precision_topk.item()
    recall_topk = target[torch.argsort(
        preds, dim=-1, descending=False)][:top_k].sum().float() / target.sum()
    recall_topk = recall_topk.item()

    y_true_binary = (y_true <= threshold).astype(int)
    y_pred_binary = (y_pred <= threshold).astype(int)

    mask = y_pred <= threshold
    range_mae = mean_absolute_error(y_true[mask],
                                    y_pred[mask]) if mask.sum() > 0 else np.nan

    precision = precision_score(y_true_binary, y_pred_binary, average='binary')
    recall = recall_score(y_true_binary, y_pred_binary, average='binary')
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    y_pred_auc = np.where(y_pred <= 100, y_pred / 100, 1)
    rocauc = roc_auc_score((y_true <= threshold), 1-y_pred_auc)
    result = {
        'mae': round(mae, 4),
        'range_mae': round(range_mae, 4),
        'AUC': round(rocauc, 4),
        'f1': round(f1, 4),
        'pcc': round(pcc, 4),
        'spcc': round(spcc, 4),
        f'precision@{top_k}': round(precision_topk, 4),
        'GT<30': len(postive_idx)
    }
    return result


def infer_model(model, data_loader, device='cuda'):

    model.to(device)
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in tqdm(data_loader, desc='Inferencing'):
            inputs = [x.to(device) for x in inputs]
            targets = targets.to(device)

            latent_feature = model(inputs)
            predicts = model.predictor(latent_feature)

            all_preds.extend(predicts.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    return np.array(all_preds).squeeze(), np.array(all_targets).squeeze()

def get_GC_count(s: pd.Series, name):
    df = s.to_frame()
    df[f"GC_count_{name}_seq"] = (s.str.count("G") +
                                  s.str.count("C")) / s.str.len()
    return df.iloc[:, 1:]  # return feature column


def get_sense_seq_gene(s: pd.Series):
    df = s.to_frame()
    antisense_gene = s.str.replace('U', 'T')
    sense_seq_gene = []
    #
    for seq in antisense_gene:
        my_seq = Seq(seq)
        reverse = str(my_seq.reverse_complement())
        sense_seq_gene.append(reverse)
    #
    df['siRNA_sense_seq_gene'] = pd.Series(sense_seq_gene)
    return df.iloc[:, 1:]  # return feature column


if __name__ == '__main__':

    print(torch.cuda.is_available())
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    parser = argparse.ArgumentParser(description="data_input")

    parser.add_argument('--inference_csv', type=str, required=True, help='Path to the test data CSV file')
    parser.add_argument('--ckpt_path', type=str, default=30, help='path for checkpoint')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate for training')
    parser.add_argument('--bs', type=int, default=256, help='Batch size for training')
    parser.add_argument('--epoches', type=int, default=500, help='Epochs for training')
    parser.add_argument('--wd', type=float, default=1e-4, help='weigth decay for training')
    parser.add_argument('--es', type=int, default=30, help='patience for early stop')

    args = parser.parse_args()

    train_data_ = pd.read_csv('train_sample.csv')
    test_data_ = pd.read_csv(args.infrerence_csv)

    train_data_ = train_data_[train_data_['gene_target_symbol_name'].isin(['APOC3', 'XDH', 'PCSK9', 'KHK'])]
    index = train_data_[train_data_.mRNA_remaining_pct >= 101].index
    train_data_.loc[index, 'mRNA_remaining_pct'] = 100

    # round_before = pd.read_csv('lab_in_the_loop_before_rounds.csv')
    # index_before = round_before[round_before.mRNA_remaining_pct >= 101].index
    # round_before.loc[index_before, 'mRNA_remaining_pct'] = 100

    train_data_.dropna(
        subset=['siRNA_antisense_seq', 'modified_siRNA_antisense_seq_list'] +
               ['mRNA_remaining_pct'] + [
                   'siRNA_concentration', 'cell_line_donor', 'Transfection_method',
                   'Duration_after_transfection_h'
               ],
        inplace=True)

    print('train/test shape', train_data_.shape, test_data_.shape)
    columns_siRNA = [
        'siRNA_antisense_seq',
        'modified_siRNA_antisense_seq_list',
        'siRNA_sense_seq',
        'modified_siRNA_sense_seq_list'
    ]
    columns_mRNA = ['gene_target_seq']

    target_columns = [
        'siRNA_sense_seq',
        'siRNA_antisense_seq',
        'modified_siRNA_sense_seq_list',
        'modified_siRNA_antisense_seq_list',
        'siRNA_concentration',
        'cell_line_donor',
        'Transfection_method',
        'Duration_after_transfection_h',
        'gene_target_symbol_name',
        'gene_target_seq',
        'mRNA_remaining_pct'
    ]
    mRNA_embedding_orthrus = train_data_.columns[train_data_.columns.str.contains('orthrus')].values.tolist()
    thermodynamics_embedding = train_data_.columns[
        train_data_.columns.str.contains('thermodynamic_properties')].values.tolist()
    target_columns = target_columns + mRNA_embedding_orthrus
    train_data_ = train_data_[target_columns]
    test_data = test_data_[target_columns]

    train_data_['Transfection_method'] = train_data_['Transfection_method'].str.upper()
    train_data_['cell_line_donor'] = train_data_['cell_line_donor'].str.upper()

    test_data['Transfection_method'] = test_data['Transfection_method'].str.upper()
    test_data['cell_line_donor'] = test_data['cell_line_donor'].str.upper()

    cell_line_donor_mapping = pd.read_csv('cell_line_donor_mapping.csv', header=0)
    cell_line_donor_mapping = dict(zip(cell_line_donor_mapping.iloc[:, 0], cell_line_donor_mapping.iloc[:, 1]))
    Transfection_method_mapping = pd.read_csv('Transfection_method_mapping.csv', header=0)
    Transfection_method_mapping = dict(zip(Transfection_method_mapping.iloc[:, 0], Transfection_method_mapping.iloc[:, 1]))
    Duration_mapping = pd.read_csv('Duration_mapping.csv', header=0)
    Duration_mapping = dict(zip(Duration_mapping.iloc[:, 0], Duration_mapping.iloc[:, 1]))

    print('cell_line_donor_mapping:',cell_line_donor_mapping)
    print('Transfection_method_mapping:',Transfection_method_mapping)
    print('Duration_mapping:',Duration_mapping)

    train_data_.cell_line_donor = train_data_.cell_line_donor.replace(
        cell_line_donor_mapping)
    train_data_.Transfection_method = train_data_.Transfection_method.replace(
        Transfection_method_mapping)
    train_data_.Duration_after_transfection_h = train_data_.Duration_after_transfection_h.replace(
        Duration_mapping)

    test_data.cell_line_donor = test_data.cell_line_donor.replace(
        cell_line_donor_mapping)
    test_data.Transfection_method = test_data.Transfection_method.replace(
        Transfection_method_mapping)
    test_data.Duration_after_transfection_h = test_data.Duration_after_transfection_h.replace(
        Duration_mapping)

    # Create vocabulary
    tokenizer = GenomicTokenizer(ngram=1, stride=1)

    with open('genomic_vocab.json', 'r') as f:
        vocab_list = json.load(f)

    vocab = GenomicVocab(vocab_list)
    print('Load vocab.json successfully!')

    max_len_siRNA = 28

    train_data, val_data = train_test_split(train_data_, test_size=0.3, random_state=24)

    train_dataset = SiRNADataset2(train_data, columns_siRNA, vocab, tokenizer, max_len_siRNA,
                                  columns_mRNA, mRNA_embedding_orthrus, thermodynamics_embedding)
    val_dataset = SiRNADataset2(val_data, columns_siRNA, vocab, tokenizer, max_len_siRNA,
                                columns_mRNA, mRNA_embedding_orthrus, thermodynamics_embedding)

    train_loader = DataLoader(train_dataset, batch_size=128, num_workers=4, drop_last=True, shuffle=True,
                              pin_memory=True, persistent_workers=True, prefetch_factor=4)
    val_loader = DataLoader(val_dataset, batch_size=128, num_workers=4, pin_memory=True, shuffle=False,
                            persistent_workers=True, prefetch_factor=4)

    feature_names = ['concentration', 'cell_line', 'transfection_method', 'duration', 'mRNA']

    n = len(feature_names)
    combinations = (
        [[True] * n]
    )

    results = []
    for combo in combinations:
        feature_mask = dict(zip(feature_names, combo))

        print("Current feature mask:", feature_mask)
        model = SiRNAModel(
            vocab_size=len(vocab.itos),
            embed_dim=256,
            cell_line_size=len(cell_line_donor_mapping),
            method_size=len(Transfection_method_mapping),
            duration_size=len(Duration_mapping),
            hidden_dim=256,
            feature_mask=feature_mask
        )

        criterion = nn.L1Loss()

        lr = args.lr
        wd = args.wd
        bs = args.bs
        ckpt = args.ckpt_path

        pairwise_optimizer = optim.Adam([{
            'params': model.encoder.parameters(),
            'lr': lr * 0.9,
            'weight_decay': wd
        }, {
            'params': model.transform.parameters(),
            'lr': lr * 0.9,
            'weight_decay': wd
        }, {
            'params': model.decoder.parameters(),
            'lr': lr * 0.9,
            'weight_decay': wd
        }])

        predict_optimizer = optim.Adam([{
            'params': model.encoder.parameters(),
            'lr': lr * 0.1,
            'weight_decay': wd
        }, {
            'params': model.transform.parameters(),
            'lr': lr * 0.1,
            'weight_decay': wd
        }, {
            'params': model.predictor.parameters(),
            'lr': lr * 0.9,
            'weight_decay': wd
        }])

        model.load_state_dict(torch.load(ckpt))
        print("Load Pretrain Successfully!")

        print("Load Fine-Tune Successfully!")
        test_dataset = SiRNADataset2(test_data, columns_siRNA, vocab, tokenizer, max_len_siRNA,
                                     columns_mRNA, mRNA_embedding_orthrus, thermodynamics_embedding)
        test_loader = DataLoader(test_dataset, batch_size=bs, num_workers=4, pin_memory=True, shuffle=False,
                                 persistent_workers=True, prefetch_factor=4)

        predictions, targets = infer_model(model, test_loader, device)
        results_df = pd.DataFrame({
            'true_value': targets,
            'predicted_value': predictions,
            'siRNA_sense_seq': test_data['siRNA_sense_seq'],
            'siRNA_antisense_seq': test_data['siRNA_antisense_seq'],
            'modified_siRNA_sense_seq_list': test_data['modified_siRNA_sense_seq_list'],
            'modified_siRNA_antisense_seq_list': test_data['modified_siRNA_antisense_seq_list'],
            'gene_target_symbol_name': test_data['gene_target_symbol_name'],
            'siRNA_concentration': test_data['siRNA_concentration'],
            'cell_line_donor': test_data['cell_line_donor']
        })

    results_df.to_csv("../rlt/rlt.csv", index=False)
    print('Done!')
