import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Sampler
import pandas as pd
import resource
from data_process import GenomicTokenizer, GenomicVocab, SiRNADataset2, read_data
from models import SiRNAEncoderWithAblation, ContrastiveEncoder, Decoder, Predictor, SiRNAModel
from train import train_model

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters()
                        if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


if __name__ == '__main__':

    print(torch.cuda.is_available())
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_data_path = 'train_data_path.csv'
    test_data_path = 'test_data_path.csv'

    train_data, test_data, columns_siRNA, vocab, tokenizer, max_len_siRNA, columns_mRNA, mRNA_embedding_orthrus, thermodynamics_embedding, cell_line_donor_mapping, Transfection_method_mapping, Duration_mapping = read_data(train_data_path, test_data_path)

    train_dataset = SiRNADataset2(train_data, columns_siRNA, vocab, tokenizer, max_len_siRNA,
                                  columns_mRNA, mRNA_embedding_orthrus, thermodynamics_embedding)
    val_dataset = SiRNADataset2(test_data, columns_siRNA, vocab, tokenizer, max_len_siRNA,
                                columns_mRNA, mRNA_embedding_orthrus, thermodynamics_embedding)

    train_loader = DataLoader(train_dataset, batch_size=256, num_workers=4, drop_last=True, shuffle=True,
                              pin_memory=True, persistent_workers=True, prefetch_factor=4)
    val_loader = DataLoader(val_dataset, batch_size=256, num_workers=4, pin_memory=True, shuffle=False,
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

        lr = 5e-5
        wd = 1e-4

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

        final_score, best_final_score, targets, predicts, final_epoch = train_model(
            model, train_loader, val_loader, criterion,
            num_epochs=500, device=device, patience=30
        )
        best_final_score_prefixed = {'best_' + key: value for key, value in best_final_score.items()}
        final_epoch = {'epoch': final_epoch}

        result = feature_mask.copy()
        result.update(final_score)
        result.update(best_final_score_prefixed)
        result.update(final_epoch)

        results.append(result)

        del model

    # 最终保存为 DataFrame
    results_df = pd.DataFrame(results)
    results_df.to_csv("Performance.csv", index=False)
