import torch
from cgnn.config import set_config, get_config, set_initial_config
from cgnn.utils import create_data_loaders, SlidingWindowDataset
from cgnn.mtad_gat import MTAD_GAT
from cgnn.prediction import Predictor


def predict_and_evaluate(model_config, train_array, test_array, anomaly_label_array, progress_callback=None, save_output=True):
    model_path = f"trained_models_temp/{model_config['dataset']}_{model_config['id']}"

    print(f'Using model from {model_path}')
    set_initial_config()
    set_config(model_config)
    model_config = get_config()
    print(f'Model config: {model_config}')

    window_size = int(model_config['lookback'])
    batch_size = int(model_config['bs'])
    val_split = float(model_config['val_split'])
    shuffle_dataset = (model_config['shuffle_dataset'] == 'True')

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

    train_dataset = SlidingWindowDataset(x_train, window_size, target_dims)
    test_dataset = SlidingWindowDataset(x_test, window_size, target_dims)

    model = MTAD_GAT(
        n_features,
        window_size,
        out_dim,
        kernel_size=int(model_config['kernel_size']),
        use_gatv2=(model_config['use_gatv2'] == 'True'),
        feat_gat_embed_dim=model_config['feat_gat_embed_dim'],
        time_gat_embed_dim=model_config['time_gat_embed_dim'],
        gru_n_layers=int(model_config['gru_n_layers']),
        gru_hid_dim=int(model_config['gru_hid_dim']),
        forecast_n_layers=int(model_config['fc_n_layers']),
        forecast_hid_dim=int(model_config['fc_hid_dim']),
        dropout=float(model_config['dropout']),
        alpha=float(model_config['alpha'])
    )

    device = "cuda" if (model_config['use_cuda'] == 'True') and torch.cuda.is_available() else "cpu"
    model.load_state_dict(torch.load(model_path+'/model.pt', map_location=device))
    model.to(device)

    # # Some suggestions for POT args
    # level_q_dict = {
    #     "SMAP": (0.90, 0.005),
    #     "MSL": (0.90, 0.001),
    #     "SMD-1": (0.9950, 0.001),
    #     "SMD-2": (0.9925, 0.001),
    #     "SMD-3": (0.9999, 0.001)
    # }
    # level, q = level_q_dict[key]
    # reg_level_dict = {"SMAP": 0, "MSL": 0, "SMD-1": 1, "SMD-2": 1, "SMD-3": 1}
    # key = "SMD-" + model_config['group'][0] if model_config['dataset'] == "SMD" else model_config['dataset']
    # reg_level = reg_level_dict[key]

    level = model_config['level']
    q = model_config['q']
    reg_level = model_config['reg_level']

    prediction_args = {
        'dataset': model_config['dataset'],
        "target_dims": target_dims,
        'scale_scores': (model_config['scale_scores'] == 'True'),
        "level": level,
        "q": q,
        'dynamic_pot': (model_config['dynamic_pot'] == 'True'),
        "use_mov_av": (model_config['use_mov_av'] == 'True'),
        "gamma": int(model_config['gamma']),
        "reg_level": reg_level,
        "save_path": model_path,
        "progress_callback": progress_callback
    }

    print(prediction_args)

    summary_file_name = "model_evaluation.json"

    label = anomaly_label_array[window_size:].flatten() if anomaly_label_array is not None else None
    predictor = Predictor(model, window_size, n_features, prediction_args, summary_file_name=summary_file_name)
    return predictor.predict_anomalies(x_train, x_test, label, save_output=save_output)
