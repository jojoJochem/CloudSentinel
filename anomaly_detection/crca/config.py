import os
import json

config_file_path = "config.json"


def get_config():
    """
    Retrieve current configuration settings from config.json.

    Returns:
    - dict: Current configuration settings.
    """
    with open(config_file_path, 'r') as file:
        config = json.load(file)
    return config


def set_config(new_config=None):
    """
    Update configuration settings in config.json with new values.

    Args:
    - new_config (dict): Dictionary containing new configuration settings.

    Writes updated settings to config.json.
    """
    config = get_config()
    if new_config:
        config.update(new_config)
    with open(config_file_path, 'w') as file:
        json.dump(config, file, indent=2)


def set_initial_config():
    """
    Initialize configuration settings based on environment variables and write to config.json.

    Environment Variables Used:
    - ARG_ETA: Learning rate (int, default: 10)
    - ARG_ITERATIONS: Number of iterations (int, default: 1)
    - ARG_GAMMA: Gamma value (float, default: 0.25)
    - EPOCHS: Number of epochs (int, default: 500)
    - BATCH_SIZE: Batch size (int, default: 50)
    - LR: Learning rate (float, default: 1e-3)
    - X_DIMS: Dimensionality of X (int, default: 1)
    - Z_DIMS: Dimensionality of Z (int, default: 1)
    - OPTIMIZER: Optimization algorithm (str, default: "Adam")
    - GRAPH_THRESHOLD: Threshold for graph (float, default: 0.3)
    - TAU_A: Tau A value (float, default: 0.0)
    - LAMBDA_A: Lambda A value (float, default: 0.0)
    - C_A: C A value (int, default: 1)
    - USE_A_CONNECT_LOSS: Use A connect loss (int, default: 0)
    - USE_A_POSITIVER_LOSS: Use A positiver loss (int, default: 0)
    - SEED: Random seed (int, default: 42)
    - ENCODER_HIDDEN: Encoder hidden units (int, default: 64)
    - DECODER_HIDDEN: Decoder hidden units (int, default: 64)
    - TEMP: Temperature (float, default: 0.5)
    - K_MAX_ITER: Maximum iterations for K (float, default: 1e2)
    - ENCODER: Encoder type (str, default: "mlp")
    - DECODER: Decoder type (str, default: "mlp")
    - NO_FACTOR: Use factor (bool, default: False)
    - ENCODER_DROPOUT: Encoder dropout rate (float, default: 0.0)
    - DECODER_DROPOUT: Decoder dropout rate (float, default: 0.0)
    - H_TOL: Tolerance for H (float, default: 1e-8)
    - LR_DECAY: Learning rate decay (int, default: 200)
    - GAMMA: Gamma value (float, default: 1.0)
    - PRIOR: Use prior (bool, default: False)

    Writes initial settings to config.json.
    """
    config = {
        "arg_eta": int(os.getenv("ARG_ETA", 10)),
        "arg_iterations": int(os.getenv("ARG_ITERATIONS", 1)),
        "arg_gamma": float(os.getenv("ARG_GAMMA", 0.25)),
        "epochs": int(os.getenv("EPOCHS", 500)),
        "batch_size": int(os.getenv("BATCH_SIZE", 50)),
        "lr": float(os.getenv("LR", 1e-3)),
        "x_dims": int(os.getenv("X_DIMS", 1)),
        "z_dims": int(os.getenv("Z_DIMS", 1)),
        "optimizer": os.getenv("OPTIMIZER", "Adam"),
        "graph_threshold": float(os.getenv("GRAPH_THRESHOLD", 0.3)),
        "tau_A": float(os.getenv("TAU_A", 0.0)),
        "lambda_A": float(os.getenv("LAMBDA_A", 0.0)),
        "c_A": int(os.getenv("C_A", 1)),
        "use_A_connect_loss": int(os.getenv("USE_A_CONNECT_LOSS", 0)),
        "use_A_positiver_loss": int(os.getenv("USE_A_POSITIVER_LOSS", 0)),
        "seed": int(os.getenv("SEED", 42)),
        "encoder_hidden": int(os.getenv("ENCODER_HIDDEN", 64)),
        "decoder_hidden": int(os.getenv("DECODER_HIDDEN", 64)),
        "temp": float(os.getenv("TEMP", 0.5)),
        "k_max_iter": float(os.getenv("K_MAX_ITER", 1e2)),
        "encoder": os.getenv("ENCODER", "mlp"),
        "decoder": os.getenv("DECODER", "mlp"),
        "no_factor": bool(os.getenv("NO_FACTOR", False)),
        "encoder_dropout": float(os.getenv("ENCODER_DROPOUT", 0.0)),
        "decoder_dropout": float(os.getenv("DECODER_DROPOUT", 0.0)),
        "h_tol": float(os.getenv("H_TOL", 1e-8)),
        "lr_decay": int(os.getenv("LR_DECAY", 200)),
        "gamma": float(os.getenv("GAMMA", 1.0)),
        "prior": bool(os.getenv("PRIOR", False))
    }
    with open(config_file_path, 'w') as file:
        json.dump(config, file, indent=2)
