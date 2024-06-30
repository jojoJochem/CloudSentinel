import json
from datetime import datetime
import torch.nn as nn
import torch
import os

from cgnn.config import set_config, get_config
from cgnn.utils import SlidingWindowDataset, create_data_loaders
from cgnn.mtad_gat import MTAD_GAT
from cgnn.training import Trainer


def train(dataset_config, train_array, test_array, anomaly_label_array, progress_callback=None):
    id = datetime.now().strftime("%d%m%Y_%H%M%S")
    set_config(dataset_config)
    config = get_config()
    config['id'] = id

    n_epochs = config['epochs']
    window_size = config['lookback']
    batch_size = config['bs']
    init_lr = config['init_lr']
    val_split = config['val_split']
    shuffle_dataset = config['shuffle_dataset']
    use_cuda = config['use_cuda']
    print_every = config['print_every']
    log_tensorboard = config['log_tensorboard']
    print(config)

    save_path = f"trained_models_temp/{config['dataset']}_{id}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    log_dir = f'{save_path}/logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    x_train = torch.from_numpy(train_array).float()
    x_test = torch.from_numpy(test_array).float()
    n_features = x_train.shape[1]

    target_dims = None
    out_dim = n_features

    train_dataset = SlidingWindowDataset(x_train, window_size, target_dims)
    test_dataset = SlidingWindowDataset(x_test, window_size, target_dims)

    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, batch_size, val_split, shuffle_dataset, test_dataset=test_dataset
    )

    model = MTAD_GAT(
        n_features,
        window_size,
        out_dim,
        kernel_size=config['kernel_size'],
        use_gatv2=config['use_gatv2'],
        feat_gat_embed_dim=config['feat_gat_embed_dim'],
        time_gat_embed_dim=config['time_gat_embed_dim'],
        gru_n_layers=config['gru_n_layers'],
        gru_hid_dim=config['gru_hid_dim'],
        forecast_n_layers=config['fc_n_layers'],
        forecast_hid_dim=config['fc_hid_dim'],
        dropout=config['dropout'],
        alpha=config['alpha']
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config['init_lr'])
    forecast_criterion = nn.MSELoss()

    args_summary = json.dumps(config, indent=2)

    trainer = Trainer(
        model,
        optimizer,
        window_size,
        n_features,
        target_dims,
        n_epochs,
        batch_size,
        init_lr,
        forecast_criterion,
        use_cuda,
        save_path,
        print_every,
        log_tensorboard,
        args_summary,
        progress_callback
    )

    trainer.fit(train_loader, val_loader)

    # Check test loss
    test_loss = trainer.evaluate(test_loader)
    print(f"Test forecast loss: {test_loss[0]:.5f}")
    print(f"Test total loss: {test_loss[1]:.5f}")

    trainer.load(f"{save_path}/model.pt")

    print(anomaly_label_array) if anomaly_label_array is not None else None
    label = anomaly_label_array[window_size:] if anomaly_label_array is not None else None

    print(x_train, x_test, label)
    print(config)

    with open(f"{save_path}/model_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(trainer.losses)

    return trainer, config
