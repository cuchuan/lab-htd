import argparse
import os
import time

import scipy.io as sio
import torch

from CL_Train import train, eval, select_best
from PyramidSSM import print_backend_status

os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Auto-select device: use GPU when available, otherwise fall back to CPU
# (e.g. Windows without the CUDA extensions built).
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Per-dataset optimal hyper-parameters. `band` is NOT stored here because it is
# read automatically from the .mat file. Each dataset keeps its own patch_size / m
# so the released pre-trained weights can be evaluated directly.
DATASET_REGISTRY = {
    "Sandiego": {"patch_size": 11, "m": 30, "test_load_weight": "ckpt_159_.pt"},
    "Sandiego2": {"patch_size": 13, "m": 5, "test_load_weight": "ckpt_12_.pt"},
    "abu-airport-2": {"patch_size": 11, "m": 5, "test_load_weight": "ckpt_187_.pt"},
    "abu-beach-4": {"patch_size": 5, "m": 15, "test_load_weight": "ckpt_180_.pt"},
}

# Defaults for datasets that are not in the registry (most common values).
DEFAULT_PATCH_SIZE = 11
DEFAULT_M = 5
DATASET_DIR = "datasets"

# ---------------------------------------------------------------------------
# Default run configuration (used when NO command-line arguments are given).
# Just edit these two values and run `python main.py` directly (e.g. from an
# IDE "Run" button). Passing --dataset / --state on the command line overrides
# whatever is set here.
# ---------------------------------------------------------------------------
DEFAULT_DATASET = "Sandiego"
DEFAULT_STATE = "eval"  # one of: train / eval / select_best


def read_band_from_mat(mat_path):
    '''
    Read the number of spectral bands from a .mat file. The hyperspectral cube is
    stored under the key `data` with shape (height, width, band).
    '''
    mat = sio.loadmat(mat_path)
    if "data" not in mat:
        raise KeyError(
            "The .mat file %s does not contain a 'data' key." % mat_path
        )
    return int(mat["data"].shape[-1])


def build_config(dataset, state):
    '''
    Assemble the full model configuration from a dataset name and a run state.
    - path is composed from the dataset name
    - band is auto-read from the .mat file
    - patch_size / m come from the registry (or defaults for unknown datasets)
    '''
    path = os.path.join(DATASET_DIR, dataset + ".mat")
    if not os.path.exists(path):
        raise FileNotFoundError(
            "Dataset file not found: %s. Put <dataset>.mat under the '%s' folder."
            % (path, DATASET_DIR)
        )

    band = read_band_from_mat(path)

    dataset_params = DATASET_REGISTRY.get(dataset, {})
    patch_size = dataset_params.get("patch_size", DEFAULT_PATCH_SIZE)
    m = dataset_params.get("m", DEFAULT_M)
    test_load_weight = dataset_params.get("test_load_weight", None)

    if test_load_weight is None and state in ("eval",):
        raise ValueError(
            "No 'test_load_weight' registered for dataset '%s'. Train it first "
            "(state=train), then evaluate a specific checkpoint." % dataset
        )

    return {
        "state": state,
        "epoch": 200,
        "band": band,
        "batch_size": 80,
        "seed": 1,
        "channel": 16,
        "lr": 1e-4,
        "multiplier": 2.,
        "epision": 10,
        "grad_clip": 1.,
        "device": DEVICE,
        "training_load_weight": None,
        "save_dir": "./models/",
        "test_load_weight": test_load_weight,
        "patch_size": patch_size,
        "m": m,
        "state_size": 16,
        "layer": 1,
        "delta": 0.1,
        "dataset": dataset,
        "path": path,
    }


def run(dataset, state):
    config = build_config(dataset, state)
    print_backend_status(config["device"])
    print(
        "[HTD-Mamba] dataset=%s | state=%s | band=%d | patch_size=%d | m=%d"
        % (dataset, state, config["band"], config["patch_size"], config["m"])
    )
    if state == "train":
        train(config)
    elif state == "select_best":
        select_best(config)
    else:
        eval(config)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified entry point for HTD-Mamba. Pick a dataset and a mode."
    )
    parser.add_argument(
        "--dataset",
        "-d",
        default=DEFAULT_DATASET,
        help="Dataset name, e.g. Sandiego, Sandiego2, abu-airport-2, abu-beach-4. "
             "The corresponding <dataset>.mat must exist under the datasets folder. "
             "If omitted, falls back to DEFAULT_DATASET defined at the top of main.py.",
    )
    parser.add_argument(
        "--state",
        "-s",
        default=DEFAULT_STATE,
        choices=["train", "eval", "select_best"],
        help="Run mode: train / eval / select_best. If omitted, falls back to "
             "DEFAULT_STATE defined at the top of main.py.",
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    start = time.perf_counter()
    run(args.dataset, args.state)
    end = time.perf_counter()
    print('time is %s' % (end - start))
