import os
import json

config_file_path = "config.json"


def set_initial_config():
    """
    Sets initial configuration parameters based on environment variables or defaults.
    Writes the configuration to 'config.json' file.

    Environment variables:
    - DATASET: Dataset name
    - GROUP: Group identifier
    - DATA_DIM: Dimension of data
    - LOOKBACK: Lookback window size
    - SPEC_RES: Special resolution flag
    - KERNEL_SIZE: Size of kernel for 1D conv layer
    - USE_GATV2: Flag to use GATV2 layers
    - FEAT_GAT_EMBED_DIM: Feature embedding dimension for GAT layers
    - TIME_GAT_EMBED_DIM: Time embedding dimension for GAT layers
    - GRU_N_LAYERS: Number of layers in GRU
    - GRU_HID_DIM: Hidden dimension size in GRU
    - FC_N_LAYERS: Number of layers in Fully Connected network
    - FC_HID_DIM: Hidden dimension size in Fully Connected network
    - ALPHA: Alpha parameter
    - EPOCHS: Number of epochs for training
    - VAL_SPLIT: Validation split ratio
    - BS: Batch size
    - INIT_LR: Initial learning rate
    - SHUFFLE_DATASET: Flag to shuffle dataset
    - DROPOUT: Dropout rate
    - USE_CUDA: Flag to use CUDA
    - PRINT_EVERY: Interval for printing progress
    - LOG_TENSORBOARD: Flag to log TensorBoard
    - SCALE_SCORES: Flag to scale scores
    - USE_MOV_AV: Flag to use moving average
    - GAMMA: Gamma parameter
    - LEVEL: Level parameter
    - Q: Q parameter
    - REG_LEVEL: Regularization level
    - DYNAMIC_POT: Flag to use dynamic potential
    - COMMENT: Any additional comments
    """
    config = {
        # -- Data params ---
        "dataset": os.getenv("DATASET", ""),
        "group": os.getenv("GROUP", ""),
        "data_dim": os.getenv("DATA_DIM", None),
        "lookback": int(os.getenv("LOOKBACK", 100)),
        # "normalize": os.getenv("NORMALIZE", True),  # Uncomment if needed
        "spec_res": bool(os.getenv("SPEC_RES", False)),
        # -- Model params ---
        # 1D conv layer
        "kernel_size": int(os.getenv("KERNEL_SIZE", 7)),
        # GAT layers
        "use_gatv2": bool(os.getenv("USE_GATV2", True)),
        "feat_gat_embed_dim": int(os.getenv("FEAT_GAT_EMBED_DIM", None)),
        "time_gat_embed_dim": int(os.getenv("TIME_GAT_EMBED_DIM", None)),
        # GRU layer
        "gru_n_layers": int(os.getenv("GRU_N_LAYERS", 1)),
        "gru_hid_dim": int(os.getenv("GRU_HID_DIM", 150)),
        # Forecasting Model
        "fc_n_layers": int(os.getenv("FC_N_LAYERS", 3)),
        "fc_hid_dim": int(os.getenv("FC_HID_DIM", 150)),
        # Other
        "alpha": float(os.getenv("ALPHA", 0.2)),
        # --- Train params ---
        "epochs": int(os.getenv("EPOCHS", 10)),
        "val_split": float(os.getenv("VAL_SPLIT", 0.1)),
        "bs": int(os.getenv("BS", 256)),
        "init_lr": float(os.getenv("INIT_LR", 1e-3)),
        "shuffle_dataset": bool(os.getenv("SHUFFLE_DATASET", True)),
        "dropout": float(os.getenv("DROPOUT", 0.4)),
        "use_cuda": bool(os.getenv("USE_CUDA", True)),
        "print_every": int(os.getenv("PRINT_EVERY", 1)),
        "log_tensorboard": bool(os.getenv("LOG_TENSORBOARD", True)),
        # --- Predictor params ---
        "scale_scores": bool(os.getenv("SCALE_SCORES", False)),
        "use_mov_av": bool(os.getenv("USE_MOV_AV", False)),
        "gamma": float(os.getenv("GAMMA", 1)),
        "level": float(os.getenv("LEVEL", 0.90)),
        "q": float(os.getenv("Q", 0.001)),
        "reg_level": float(os.getenv("REG_LEVEL", 1)),
        "dynamic_pot": bool(os.getenv("DYNAMIC_POT", False)),
        # --- Other ---
        "comment": os.getenv("COMMENT", "")
    }

    with open(config_file_path, 'w') as file:
        json.dump(config, file, indent=2)


def get_config():
    """
    Retrieves the current configuration from 'config.json' file.

    Returns:
    - Dictionary containing the current configuration.
    """
    with open(config_file_path, 'r') as file:
        config = json.load(file)
    return config


def set_config(new_config=None):
    """
    Updates the configuration parameters in 'config.json' file.

    Args:
    - new_config: Optional. Dictionary containing new configuration parameters.

    If new_config is provided, it updates the existing configuration parameters.
    """
    config = get_config()
    if new_config:
        config.update(new_config)
    with open(config_file_path, 'w') as file:
        json.dump(config, file, indent=2)
