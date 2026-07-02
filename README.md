# Training-Free Variational Data assimilation with Pangu-Weather

This repository implements a data assimilation approach for the [Pangu-Weather](https://github.com/198808xc/Pangu-Weather) model. Given partial or uncertain atmospheric observations, the method uses L-BFGS optimization to find an initial atmospheric state that satisfies prior knowledge constraints while remaining close to the model's climatological distribution.

## Structure

```
DA_pangu/
├── data/
│   └── era5_dataloader/        # ERA5 data loading and normalization utilities
├── models/
│   └── pangu/
│       ├── prio_knowledge_training_single_gpu.py   # Main training/assimilation script
│       ├── utils.py                                # Metrics and loss utilities
│       ├── config.yaml                             # Training configuration
│       ├── execute.sh                              # Run script with default parameters
│       ├── weatherlearn_utils/                     # Patch embedding utilities (from WeatherLearn)
│       └── ...
├── requirements.txt
└── README.md
```

## Setup

Install [uv](https://docs.astral.sh/uv/) and then:

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Data

Normalization statistics (`pangu_norm_stats2_with_w.pt`) are computed from ERA5 data using `data/era5_dataloader/find_norm_tensors.py`. The normalization statistics may partially originate from [ArchesWeather](https://github.com/gcouairon/ArchesWeather). The Pangu-Weather ONNX model weights must be downloaded separately from the [official release](https://github.com/198808xc/Pangu-Weather).

## Usage

```bash
cd models/pangu
bash execute.sh
```

Parameters can be overridden via environment variables:

```bash
PRIOR_DIM="[0,3]" REG_PARAM=1e7 LBFGSSTEP=50 bash execute.sh
```

Key arguments for `prio_knowledge_training_single_gpu.py`:

| Argument | Default | Description |
|---|---|---|
| `--model_path` | required | Path to Pangu-Weather ONNX model |
| `--data_path` | required | Path to ERA5 NetCDF file |
| `--prior_known_dim` | `[1,2]` | Which atmospheric dimensions are treated as known |
| `--reg_param` | `1e8` | Regularization strength |
| `--LBFGSsteps` | `15` | Number of L-BFGS inner steps |
| `--known_atmosphere` | `None` | Index of known atmosphere mask |

## Experiment tracking

The code uses [Weights & Biases](https://wandb.ai/). Set your API key before running:

```bash
export WANDB_API_KEY=<your_key>
```

The project name is configured in `config.yaml` under `wandb.project`.

## Citations

If you use this code, please cite the following works:

**Pangu-Weather** (base model):
```bibtex
@article{bi2023accurate,
  title={Accurate medium-range global weather forecasting with 3D neural networks},
  author={Bi, Kaifeng and Xie, Lingxi and Zhang, Hengheng and Chen, Xin and Gu, Xiaotao and Tian, Qi},
  journal={Nature},
  volume={619},
  pages={533--538},
  year={2023},
  publisher={Nature Publishing Group}
}
```
Also 
**WeatherLearn** (patch embedding utilities in `weatherlearn_utils/`):

which we have used for the dataloader and wand to thank for their good work.
```
https://github.com/lizhuoq/WeatherLearn
```
