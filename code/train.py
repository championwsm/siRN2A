import torch
import numpy as np
from tqdm import tqdm

def train_model(model,
                train_loader,
                val_loader,
                criterion,
                num_epochs=50,
                device='cuda',
                patience=10):
    model.to(device)
    best_score_metric = -float('inf')
    best_score = -float('inf')
    best_model = None
    counter = 0
    best_val_loss = float('inf')

    loss_train_total = []
    loss_valid_total = []
    score_total = []
    import time

    for epoch in range(num_epochs):
        s1 = time.time()
        model.train()
        train_loss = 0

        print(f"\nEpoch {epoch + 1}/{num_epochs} ----------------")
        for inputs1, targets1 in train_loader:
            for inputs2, targets2 in tqdm(train_loader):
                inputs1 = [x.to(device) for x in inputs1]
                inputs2 = [x.to(device) for x in inputs2]
                targets1 = targets1.to(device)
                targets2 = targets2.to(device)

                pairwise_optimizer.zero_grad()
                encoder1 = model(inputs1)
                encoder2 = model(inputs2)

                concat_feature = torch.cat(
                    [encoder1, encoder2, encoder1 - encoder2], dim=1)
                predict_diff = model.decoder(concat_feature)
                true_diff = targets1 - targets2

                loss_ae_train = criterion(predict_diff.squeeze(),
                                          true_diff.squeeze())
                loss_ae_train.backward()
                pairwise_optimizer.step()
                train_loss += loss_ae_train.item()
            break

        train_loss /= len(train_loader)

        print(f"Epoch {epoch + 1} Train Loss: {train_loss:.6f}")

        # Predictor Update
        if (epoch + 1) % 1 == 0:  # 每个 epoch 更新 predictor
            for inputs, targets in tqdm(train_loader):
                inputs = [x.to(device) for x in inputs]
                targets = targets.to(device)
                predict_optimizer.zero_grad()
                latent_feature = model(inputs)
                predicts = model.predictor(latent_feature)
                loss_pred_train = criterion(predicts.squeeze(),
                                            targets.squeeze())
                loss_pred_train.backward()
                predict_optimizer.step()

            print("Updated predictor")
        model.eval()
        val_loss = 0
        val_preds = []
        val_targets = []
        s2 = time.time()
        # print('training time:', s2 - s1)
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader,
                                        desc=f'Validating epoch {epoch + 1}'):
                inputs = [x.to(device) for x in inputs]
                targets = targets.to(device)
                latent_feature = model(inputs)
                predicts = model.predictor(latent_feature)
                loss = criterion(predicts.squeeze(), targets.squeeze())
                val_loss += loss.item()

                val_preds.extend(predicts.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())

        val_loss /= len(val_loader)
        # print('eval time:', time.time() - s2)
        print(f"Epoch {epoch + 1} Validation Loss: {val_loss:.6f}")

        # scheduler.step(val_loss)

        loss_train_total.append(train_loss)
        loss_valid_total.append(val_loss)

        val_preds = np.array(val_preds).squeeze()
        val_targets = np.array(val_targets).squeeze()
        score = calculate_metrics(val_targets, val_preds)
        print('score: ', score)
        score_total.append(score)

        # score = score['spcc']
        # if score > best_score:
        #     best_score = score
        score_spcc = score['spcc']
        if score_spcc > best_score_metric:
            best_score_metric = score_spcc
            best_score = score
            best_model = model.state_dict().copy()
            print(f'New best model found with socre: {best_score_metric:.4f}')
            torch.save(best_model, 'checkpoint/state_dict_model_26_28_modis_x_siRN2A.pth')
            counter = 1
        else:
            counter += 1

        # if val_loss < 14:
        #     break
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # counter = 1
            print(f'Best Validation Loss:{best_val_loss}')
        # else:
        #     counter += 1
        print(f'counter = {counter}')
        if counter >= patience or epoch == (num_epochs - 1):
            print('Out of Patience! BREAK!')
            print(f'Best Score: {best_score_metric:.4f}')
            print(f'New best model found with socre: {best_score_metric:.4f}')
            score_total.append(best_score)
            break

    return score_total[-2], score_total[-1], targets.cpu().numpy().squeeze(), predicts.cpu(
    ).numpy().squeeze(), epoch