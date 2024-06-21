# config.py
import os
import json

config_file_path = "config.json"


def set_initial_config():
    config = {
        # -- Data params ---
        "dataset": os.getenv("DATASET", ""),
        "group": os.getenv("GROUP", ""),
        "data_dim": os.getenv("DATA_DIM", ),
        "lookback": os.getenv("LOOKBACK", 100),
        # "normalize": os.getenv("NORMALIZE", True),
        "spec_res": os.getenv("SPEC_RES", False),
        # -- Model params ---
        # 1D conv layer
        "kernel_size": os.getenv("KERNEL_SIZE", 7),
        # GAT layers
        "use_gatv2": os.getenv("USE_GATV2", True),
        "feat_gat_embed_dim": os.getenv("FEAT_GAT_EMBED_DIM", None),
        "time_gat_embed_dim": os.getenv("TIME_GAT_EMBED_DIM", None),
        # GRU layer
        "gru_n_layers": os.getenv("GRU_N_LAYERS", 1),
        "gru_hid_dim": os.getenv("GRU_HID_DIM", 150),
        # Forecasting Model
        "fc_n_layers": os.getenv("FC_N_LAYERS", 3),
        "fc_hid_dim": os.getenv("FC_HID_DIM", 150),
        # Other
        "alpha": os.getenv("ALPHA", 0.2),
        # --- Train params ---
        "epochs": os.getenv("EPOCHS", 10),
        "val_split": os.getenv("VAL_SPLIT", 0.1),
        "bs": os.getenv("BS", 256),
        "init_lr": os.getenv("INIT_LR", 1e-3),
        "shuffle_dataset": os.getenv("SHUFFLE_DATASET", True),
        "dropout": os.getenv("DROPOUT", 0.4),
        "use_cuda": os.getenv("USE_CUDA", True),
        "print_every": os.getenv("PRINT_EVERY", 1),
        "log_tensorboard": os.getenv("LOG_TENSORBOARD", True),
        # --- Predictor params ---
        "scale_scores": os.getenv("SCALE_SCORES", False),
        "use_mov_av": os.getenv("USE_MOV_AV", False),
        "gamma": os.getenv("GAMMA", 1),
        "level": os.getenv("LEVEL", 0.90),
        "q": os.getenv("Q", 0.001),
        "reg_level": os.getenv("REG_LEVEL", 1),
        "dynamic_pot": os.getenv("DYNAMIC_POT", False),
        # --- Other ---
        "comment": os.getenv("COMMENT", "")
    }
    with open(config_file_path, 'w') as file:
        json.dump(config, file, indent=2)


def get_config():
    with open(config_file_path, 'r') as file:
        config = json.load(file)
    return config


def set_config(new_config=None):
    config = get_config()
    if new_config:
        config.update(new_config)
    with open(config_file_path, 'w') as file:
        json.dump(config, file, indent=2)
