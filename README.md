## Installation
- CUDA 11.7 or 11.8
  - Make sure `/usr/local/cuda-11.7` exists. If not, you can install it from NVIDIA DEVELOPER ([CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive)). For example, Ubuntu 18.04 x86_64
    - `wget https://developer.download.nvidia.com/compute/cuda/11.7.1/local_installers/cuda_11.7.1_515.65.01_linux.run`
    - `sudo sh cuda_11.7.1_515.65.01_linux.run`
  
    Note that if you use Ubuntu 24.04, perhaps the gcc version is too high to install CUDA 11.7. So you should degrade the gcc version at first like this:
    - `sudo apt install gcc-9 g++-9`
    - `sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 100 `
    - `gcc -v`
  - See `nvcc -V` is `cuda_11.7`. If not, you should modify the `.bashrc` like this:
    - `vim ~/.bashrc` -> `i`, and add the following to the end
    
      export CUDA_HOME=/usr/local/cuda-11.7
    
      export PATH=$PATH:/usr/bin:/bin
    
      export PATH=/usr/local/cuda-11.7/bin${PATH:+:${PATH}}
    
      export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
    - `esc`->`:wq`->`source ~/.bashrc`, and see `nvcc -V` is `cuda_11.7`.
- Python 3.10.x
  - `conda create -n htd-mamba python=3.10`
  - `conda activate htd-mamba`

- Torch 2.0.1
  - `conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia` or
  - `pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117`

- Requirements: requirements.txt
  - `cd /home/your_path/HTD-Mamba-main`
  - `pip install -r requirements.txt`

- Install ``selective_scan_cuda``
  - `cd /home/your_path/HTD-Mamba-main`
  - `pip install .`
  
- Install ``causal_conv1d``
  - `pip install --upgrade pip`
  - `pip install causal_conv1d==1.4.0`

## Installation (Windows, no CUDA compilation)
The `selective_scan_cuda` and `causal_conv1d` extensions above are **CUDA-only accelerators** that are hard to build on Windows. They are now **optional**: when they are not installed, the code automatically falls back to an **equivalent pure-PyTorch implementation** that produces the same results (only slower). The device is auto-detected, so it runs on an NVIDIA GPU if available and otherwise on CPU.

Just run the following **three commands** (Windows + NVIDIA GPU). No C++/CUDA compilation, and you do **NOT** need `pip install .` or `causal_conv1d`:

```bash
# 1) Create and activate the environment (Python 3.10 is required)
conda create -n htd-mamba python=3.10 -y && conda activate htd-mamba

# 2) Install everything, including the GPU (CUDA 11.8) build of PyTorch
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118

# 3) Run (defaults to evaluating Sandiego; edit DEFAULT_DATASET/DEFAULT_STATE in main.py or pass -d/-s)
python main.py --dataset Sandiego
```

- **CPU-only machine?** Drop the `--extra-index-url ...` part in step 2 (`pip install -r requirements.txt`); it will install CPU PyTorch and everything still runs, just slower.
- On startup the program prints a line like `[HTD-Mamba] device=cuda:0 | selective_scan_cuda=OFF(fallback) | causal_conv1d=OFF(fallback)`, so you can confirm it is using your GPU with the pure-PyTorch fallback.

## Unified entry point
All datasets now share a single entry point `main.py`. You only pass the dataset
name and the mode; `band` is auto-read from the `.mat` file, `path` is composed
from the dataset name, and the optimal `patch_size` / `m` are looked up from a
built-in registry (unknown datasets fall back to `patch_size=11`, `m=5`).

```bash
# Evaluate (default mode) on the four public datasets:
python main.py --dataset Sandiego
python main.py --dataset Sandiego2
python main.py --dataset abu-airport-2
python main.py --dataset abu-beach-4

# Train / select the best checkpoint:
python main.py --dataset Sandiego --state train
python main.py --dataset Sandiego --state select_best
```

- `--dataset` / `-d`: dataset name; the corresponding `datasets/<dataset>.mat` must exist.
- `--state` / `-s`: `eval` (default), `train`, or `select_best`.
- To use a **new dataset**, drop `datasets/<name>.mat` in place and run `python main.py -d <name> -s train`. To pin its optimal `patch_size` / `m` and a checkpoint for later `eval`, add an entry to `DATASET_REGISTRY` in `main.py`.
## Acknowledgement
This project is based on `Mamba` ([paper](https://arxiv.org/abs/2312.00752), [code](https://github.com/state-spaces/mamba)), `Vim` ([paper](https://arxiv.org/abs/2401.09417), [code](https://github.com/hustvl/Vim)). We thank the authors for their promising studies.


## Citation
If you find `HTD-Mamba` useful in your research or applications, please consider giving us a star 🌟 and citing it using the following BibTeX entry.

```bibtex
 @article{shen2025htd,
	author={Shen, Dunbin and Zhu, Xuanbing and Tian, Jiacheng and Liu, Jianjun and Du, Zhenrong and Wang, Hongyu and Ma, Xiaorui},
	journal={IEEE Transactions on Geoscience and Remote Sensing}, 
	title={HTD-Mamba: Efficient Hyperspectral Target Detection With Pyramid State Space Model}, 
	year={2025},
	month={Mar.},
	volume={63},
	note={{Art. no. 5507315}},
}
```
