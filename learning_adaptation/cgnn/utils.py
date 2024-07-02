import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
import pandas as pd
from tqdm import tqdm


class SlidingWindowDataset(Dataset):
    def __init__(self, data, window, target_dim=None, horizon=1):
        self.data = data
        self.window = window
        self.target_dim = target_dim
        self.horizon = horizon

    def __getitem__(self, index):
        x = self.data[index: index + self.window]
        y = self.data[index + self.window: index + self.window + self.horizon]
        return x, y

    def __len__(self):
        return len(self.data) - self.window


def permutation_importance(model, loader, criterion, device='cpu', progress_callback=None):
    model.eval()
    initial_losses = []

    for x, y in tqdm(loader, desc="Calculating initial loss", leave=False):
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            preds = model(x).squeeze(1)
            if preds.shape != y.shape:
                preds = preds.view_as(y)
            loss = criterion(preds, y)
        initial_losses.append(loss.item())
    initial_loss = np.mean(initial_losses)

    importance = {}
    features = loader.dataset[0][0].shape[1]
    for i in tqdm(range(features), desc="Permuting features", leave=True):
        feature_losses = []
        for batch_index, (x, y) in enumerate(tqdm(loader, desc=f"Feature {i}", leave=False)):
            original_x = x.clone()
            np.random.shuffle(x[:, :, i].numpy())  # Shuffle feature i
            x = x.to(device)
            with torch.no_grad():
                preds = model(x).squeeze(1)
                if preds.shape != y.shape:
                    preds = preds.view_as(y)
                loss = criterion(preds, y)
            feature_losses.append(loss.item())
            x[:, :, i] = original_x[:, :, i]  # Restore original data
            if progress_callback:
                progress_callback('FEATURE_IMPORTANCE', '', i, features, batch_index, len(loader))

        mean_feature_loss = np.mean(feature_losses)
        importance[i] = mean_feature_loss - initial_loss

    return importance


def create_data_loaders(train_dataset, batch_size, val_split=0.1, shuffle=True, test_dataset=None):
    train_loader, val_loader, test_loader = None, None, None
    if val_split == 0.0:
        print(f"train_size: {len(train_dataset)}")
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    else:
        dataset_size = len(train_dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(val_split * dataset_size))
        if shuffle:
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
        val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=valid_sampler)

        print(f"train_size: {len(train_indices)}")
        print(f"validation_size: {len(val_indices)}")

    if test_dataset is not None:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        print(f"test_size: {len(test_dataset)}")

    return train_loader, val_loader, test_loader


def load(model, trained_model, device="cpu"):
    """
    Loads the model's parameters from the trained model
    """
    model.load_state_dict(torch.load(trained_model, map_location=device))


def adjust_anomaly_scores(scores, dataset, is_train, lookback):
    """
    Method for MSL and SMAP where channels have been concatenated as part of the preprocessing
    :param scores: anomaly_scores
    :param dataset: name of dataset
    :param is_train: if scores is from train set
    :param lookback: lookback (window size) used in model
    """

    # Remove errors for time steps when transition to new channel (as this will be impossible for model to predict)
    if dataset.upper() not in ['SMAP', 'MSL']:
        return scores

    adjusted_scores = scores.copy()
    if is_train:
        md = pd.read_csv(f'./datasets/data/{dataset.lower()}_train_md.csv')
    else:
        md = pd.read_csv('./datasets/data/labeled_anomalies.csv')
        md = md[md['spacecraft'] == dataset.upper()]

    md = md[md['chan_id'] != 'P-2']

    # Sort values by channel
    md = md.sort_values(by=['chan_id'])

    # Getting the cumulative start index for each channel
    sep_cuma = np.cumsum(md['num_values'].values) - lookback
    sep_cuma = sep_cuma[:-1]
    buffer = np.arange(1, 20)
    i_remov = np.sort(np.concatenate((sep_cuma, np.array([i+buffer for i in sep_cuma]).flatten(),
                                      np.array([i-buffer for i in sep_cuma]).flatten())))
    i_remov = i_remov[(i_remov < len(adjusted_scores)) & (i_remov >= 0)]
    i_remov = np.sort(np.unique(i_remov))
    if len(i_remov) != 0:
        adjusted_scores[i_remov] = 0

    # Normalize each concatenated part individually
    sep_cuma = np.cumsum(md['num_values'].values) - lookback
    s = [0] + sep_cuma.tolist()
    for c_start, c_end in [(s[i], s[i+1]) for i in range(len(s)-1)]:
        e_s = adjusted_scores[c_start: c_end+1]

        e_s = (e_s - np.min(e_s))/(np.max(e_s) - np.min(e_s))
        adjusted_scores[c_start: c_end+1] = e_s

    return adjusted_scores
