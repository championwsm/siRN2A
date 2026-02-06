import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter


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
        # self.thermodynamics_embedding = thermodynamics_embedding

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

def get_GC_count(s: pd.Series, name):
    df = s.to_frame()
    df[f"GC_count_{name}_seq"] = (s.str.count("G") +
                                  s.str.count("C")) / s.str.len()
    return df.iloc[:, 1:]  # 返回特征列


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
    return df.iloc[:, 1:]  # 返回特征列

def read_data(train_data_path, test_data_path):
    train_data_ = pd.read_csv(
        train_data_path)
    index = train_data_[train_data_.mRNA_remaining_pct >= 101].index
    train_data_.loc[index, 'mRNA_remaining_pct'] = 100
    test_data = pd.read_csv(
        test_data_path)

    train_data_.dropna(
        subset=['siRNA_antisense_seq', 'modified_siRNA_antisense_seq_list'] +
               ['mRNA_remaining_pct'] + [
                   'siRNA_concentration', 'cell_line_donor', 'Transfection_method',
                   'Duration_after_transfection_h'
               ],
        inplace=True)

    print('train/test shape', train_data_.shape, test_data.shape)
    columns_siRNA = [
        'siRNA_antisense_seq',
        'modified_siRNA_antisense_seq_list',
        'siRNA_sense_seq'
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
        'mRNA_remaining_pct'
        # 'thermodynamic_properties'
    ]
    mRNA_embedding_orthrus = train_data_.columns[train_data_.columns.str.contains('orthrus')].values.tolist()
    thermodynamics_embedding = train_data_.columns[
        train_data_.columns.str.contains('thermodynamic_properties')].values.tolist()
    target_columns = target_columns + mRNA_embedding_orthrus
    train_data_ = train_data_[target_columns]
    test_data = test_data[target_columns]

    train_data_['Transfection_method'] = train_data_['Transfection_method'].str.upper()
    train_data_['cell_line_donor'] = train_data_['cell_line_donor'].str.upper()
    # train_data_['gene_target_symbol_name'] = train_data_['gene_target_symbol_name'].str.upper()

    test_data['Transfection_method'] = test_data['Transfection_method'].str.upper()
    test_data['cell_line_donor'] = test_data['cell_line_donor'].str.upper()
    # test_data['gene_target_symbol_name'] = test_data['gene_target_symbol_name'].str.upper()

    cell_line_donor_mapping = pd.read_csv(
        'cell_line_donor_mapping.csv',
        header=0)
    # cell_line_donor_mapping = cell_line_donor_mapping.set_index(0)[1].to_dict()
    cell_line_donor_mapping = dict(zip(cell_line_donor_mapping.iloc[:, 0], cell_line_donor_mapping.iloc[:, 1]))
    Transfection_method_mapping = pd.read_csv(
        'Transfection_method_mapping.csv',
        header=0)
    # Transfection_method_mapping = Transfection_method_mapping.set_index(0)[1].to_dict()
    Transfection_method_mapping = dict(
        zip(Transfection_method_mapping.iloc[:, 0], Transfection_method_mapping.iloc[:, 1]))
    Duration_mapping = pd.read_csv(
        'Duration_mapping.csv', header=0)
    # Duration_mapping = Duration_mapping.set_index(0)[1].to_dict()
    Duration_mapping = dict(zip(Duration_mapping.iloc[:, 0], Duration_mapping.iloc[:, 1]))
    # result_dict_float = {key: float(value) for key, value in result_dict.items()}

    # print('siRNA_concentration_mapping:',siRNA_concentration_mapping)
    print('cell_line_donor_mapping:', cell_line_donor_mapping)
    print('Transfection_method_mapping:', Transfection_method_mapping)
    print('Duration_mapping:', Duration_mapping)

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

    all_data = train_data_

    # Create vocabulary
    tokenizer = GenomicTokenizer(ngram=1, stride=1)

    all_tokens = []
    for col in columns_siRNA:
        for seq in all_data[col]:
            # for seq in train_data_[col]:
            if ' ' in seq:  # Modified sequence
                all_tokens.extend(seq.split())
            else:
                all_tokens.extend(tokenizer.tokenize(seq))

    vocab = GenomicVocab.create(all_tokens, max_vocab=10000, min_freq=1)
    print(len(vocab.itos))
    print(vocab.itos)

    # Find max sequence length
    max_len_siRNA = max(
        max(
            len(seq.split()) if ' ' in seq else len(tokenizer.tokenize(seq))
            for seq in all_data[col]) for col in columns_siRNA)

    train_data, test_data = train_test_split(all_data, test_size=0.3, random_state=24)
    return (train_data, test_data,
            columns_siRNA, vocab, tokenizer,
            max_len_siRNA, columns_mRNA, mRNA_embedding_orthrus,
            thermodynamics_embedding, cell_line_donor_mapping, Transfection_method_mapping, Duration_mapping)
