import torch
import torch.nn as nn


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

        # 7 可变特征 + 3 固定特征 = 10 * embed_dim 输入
        self.fc_mlp = nn.Linear(4 * embed_dim, 4 * embed_dim)
        self.fc_gru = nn.Linear(4 * embed_dim, 4 * embed_dim)       # seq的条数

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