import json
import os
import torch

from config import set_config, get_config, set_initial_config
from utils import load, create_data_loaders, SlidingWindowDataset
from mtad_gat import MTAD_GAT
from prediction import Predictor


def load_model_and_predict(test_array, dataset, save_output=True):
    if os.path.isdir(f"trained_models/{dataset}"):
        model_path = f"trained_models/{dataset}"
    else:
        raise Exception(f'Dataset "{dataset}" not available.')

    # Check that model exist
    if not os.path.isfile(f"{model_path}/model.pt"):
        raise Exception(f"<{model_path}/model.pt> does not exist.")

    model_config_path = f"{model_path}/model_config.json"

    # Get configs of model
    print(f'Using model from {model_path}')
    set_initial_config()
    with open(model_config_path, "r") as f:
        set_config(json.load(f))
    model_config = get_config()
    print(f'Model config: {model_config}')

    window_size = model_config['lookback']
    batch_size = model_config['bs']
    val_split = model_config['val_split']
    shuffle_dataset = model_config['shuffle_dataset']

    x_test = torch.from_numpy(test_array).float()
    n_features = x_test.shape[1]
    target_dims = None
    # TODO check if this is correct
    # TODO
    # TODO CHEKCHECKECHEKCHEKCHECK
    out_dim = 38

    test_dataset = SlidingWindowDataset(x_test, window_size, target_dims)
    train_loader, val_loader, test_loader = create_data_loaders(
        test_dataset, batch_size, val_split, shuffle_dataset
    )
    test_dataset = SlidingWindowDataset(x_test, window_size, target_dims)

    model = MTAD_GAT(
        n_features,
        window_size,
        out_dim,
        kernel_size=model_config['kernel_size'],
        use_gatv2=model_config['use_gatv2'],
        feat_gat_embed_dim=model_config['feat_gat_embed_dim'],
        time_gat_embed_dim=model_config['time_gat_embed_dim'],
        gru_n_layers=model_config['gru_n_layers'],
        gru_hid_dim=model_config['gru_hid_dim'],
        forecast_n_layers=model_config['fc_n_layers'],
        forecast_hid_dim=model_config['fc_hid_dim'],
        dropout=model_config['dropout'],
        alpha=model_config['alpha']
    )

    device = "cuda" if model_config['use_cuda'] and torch.cuda.is_available() else "cpu"
    load(model, f"{model_path}/model.pt", device=device)
    model.to(device)

    # # Some suggestions for POT args
    # level_q_dict = {
    #     "SMAP": (0.90, 0.005),
    #     "MSL": (0.90, 0.001),
    #     "SMD-1": (0.9950, 0.001),
    #     "SMD-2": (0.9925, 0.001),
    #     "SMD-3": (0.9999, 0.001)
    # }
    # key = "SMD-" + model_config['group'][0] if model_config['dataset'] == "SMD" else model_config['dataset']
    # if key in level_q_dict:
    #     level, q = level_q_dict[key]
    # else:
    #     level, q = (0.9950, 0.001)
    # if model_config['level'] is not None:
    #     level = model_config['level']
    # if model_config['q'] is not None:
    #     q = model_config['q']

    # # Some suggestions for Epsilon args
    # reg_level_dict = {"SMAP": 0, "MSL": 0, "SMD-1": 1, "SMD-2": 1, "SMD-3": 1}
    # key = "SMD-" + model_config['group'][0] if dataset == "SMD" else dataset
    # if key in reg_level_dict:
    #     reg_level = reg_level_dict[key]
    # else:
    #     reg_level = 1
    level = model_config['level']
    q = model_config['q']
    reg_level = model_config['reg_level']

    prediction_args = {
        'dataset': dataset,
        "target_dims": target_dims,
        'scale_scores': model_config['scale_scores'],
        "level": level,
        "q": q,
        'dynamic_pot': model_config['dynamic_pot'],
        "use_mov_av": model_config['use_mov_av'],
        "gamma": model_config['gamma'],
        "reg_level": reg_level,
        "save_path": f"{model_path}",
    }

    # load summary.txt file a json dict
    summary_file_name = f"{model_path}/model_evaluation.json"
    if os.path.isfile(summary_file_name):
        with open(summary_file_name, "r") as f:
            summary = json.load(f)
            prediction_args.update(summary)
        global_epsilon = summary['epsilon_result']['threshold']
    else:
        global_epsilon = 0.1
    print(global_epsilon)

    predictor = Predictor(model, window_size, n_features, prediction_args)
    df_test = predictor.predict_anomalies_without_labels(x_test, global_epsilon, save_output=save_output)

    return check_anomalies(df_test)


def check_anomalies(df):
    anomalies = df['A_Pred_Global'].gt(0).sum()
    total = len(df)
    percentage = anomalies / total * 100
    return percentage


if __name__ == "__main__":
    load_model_and_predict(new_dataset="SMD", model_id="25052024_210310",
                           dataset="SMD", group="1-1", save_output=True)
    load_model_and_predict(new_dataset="SMD", model_id="26052024_024540",
                           dataset="SMD", group="1-2", save_output=True)
    load_model_and_predict(new_dataset="SMD", model_id="26052024_101725",
                           dataset="SMD", group="1-7", save_output=True)
