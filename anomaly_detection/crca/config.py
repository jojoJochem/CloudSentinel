# config.py
import os
import json

config_file_path = "config.json"


def get_config():
    with open(config_file_path, 'r') as file:
        config = json.load(file)
    return config


def set_config(new_config=None):
    # Update the configuration file with new settings
    config = get_config()
    if new_config:
        config.update(new_config)
    with open(config_file_path, 'w') as file:
        json.dump(config, file)


def set_initial_config():
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
        json.dump(config, file)
