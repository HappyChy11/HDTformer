# HDTformer

This repository contains the PyTorch source code, data preparation utility,
experiment configurations, and tests required to train and evaluate HDTformer.

## Repository layout

```text
HDTformer/
├── configs/                    YAML experiment configurations
├── hdtformer/
│   ├── config.py               configuration loading and path resolution
│   ├── data.py                 feature construction, splitting, scaling, windows
│   ├── engine.py               training and validation loops
│   ├── metrics.py              MAE, RMSE, and R2
│   ├── model.py                model assembly and forward computation
│   ├── msdcd.py                MSDCD implementation
│   ├── reproducibility.py      random-seed helper
│   ├── resources.py            runtime and resource monitoring
│   ├── temporal_vector.py      TV implementation
│   └── transformer.py          Transformer blocks
├── tools/
│   ├── prepare_rv_data.py      download prices and construct RV data
│   └── run_experiment.py       train and evaluate from a YAML configuration
├── tests/                      reproducibility and tensor-shape tests
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.10 or later
- PyTorch 2.0 or later
- A CUDA-compatible GPU is optional. CPU execution is supported but slower.

The Python dependencies are listed in `requirements.txt`:

```text
numpy>=1.23
pandas>=1.5
scikit-learn>=1.2
torch>=2.0
PyYAML>=6.0
psutil>=5.9
tqdm>=4.65
```

## Installation

Clone the repository and enter the source-package directory:

```bash
git clone https://github.com/HappyChy11/HDTformer.git
cd HDTformer/HDTformer
```

Create an isolated environment and install the dependencies.

Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If a particular CUDA build of PyTorch is required, install PyTorch using the
command provided at <https://pytorch.org/get-started/locally/> before installing
the remaining requirements.

## Prepare the data

Raw one-minute spot klines are downloaded from Binance Public Data:

<https://data.binance.vision/data/spot/monthly/klines/>

Create a `Data` directory at the same level as `configs`:

```bash
mkdir Data
```

The following command downloads BNBUSDT one-minute klines and constructs the
primary rolling five-minute RV series:

```bash
python tools/prepare_rv_data.py \
  --symbol BNBUSDT \
  --start 2022-01-01T00:00:00Z \
  --end 2023-05-30T23:59:00Z \
  --window 5 \
  --output Data/RV_BNB.csv
```

Replace `BNBUSDT` and the output filename to prepare BTC, ETH, or XRP:

```bash
python tools/prepare_rv_data.py --symbol BTCUSDT --start 2022-01-01T00:00:00Z --end 2023-05-30T23:59:00Z --window 5 --output Data/RV_BTC.csv
python tools/prepare_rv_data.py --symbol ETHUSDT --start 2022-01-01T00:00:00Z --end 2023-05-30T23:59:00Z --window 5 --output Data/RV_ETH.csv
python tools/prepare_rv_data.py --symbol XRPUSDT --start 2022-01-01T00:00:00Z --end 2023-05-30T23:59:00Z --window 5 --output Data/RV_XRP.csv
```

Each generated CSV has the following schema:

```text
timestamp,realized_volatility_5T
2022-01-01T00:04:00Z,...
```

Timestamps are stored in UTC. Because RV uses a trailing five-minute window,
the first output timestamp is four minutes after the requested start time.

To use an existing dataset instead, provide a CSV with the same two columns and
change `data.path` in the YAML configuration. Relative paths are resolved from
the location of the YAML file, not from the current terminal directory.

## Verify the installation

Run the included tests from the source-package directory:

```bash
python -m unittest discover -s tests -v
```

The tests verify:

- the configured TV feature selection and output dimension;
- the dimensions of the dynamic-weight generator and basis kernels;
- MSDCD and full-model forward tensor shapes;
- causal feature construction and input scaling.

## Run an experiment

The experiment runner trains a one-step model, retains the checkpoint with the
lowest validation loss, and evaluates every horizon listed in `data.tasks`.

Run the default BTC configuration:

```bash
python tools/run_experiment.py \
  --config configs/default.yaml \
  --device auto
```

Run the BNB configuration with a specific seed and output directory:

```bash
python tools/run_experiment.py \
  --config configs/bnb_multiseed.yaml \
  --seed 12 \
  --device cuda \
  --output-dir outputs/BNB_seed12
```

Available command-line arguments are:

| Argument | Default | Description |
|---|---:|---|
| `--config` | `configs/default.yaml` | YAML configuration file |
| `--output-dir` | configuration-dependent | Directory for checkpoints and results |
| `--device` | `auto` | `auto`, `cpu`, or `cuda` |
| `--seed` | value in the YAML file | Override the experiment seed |

`--device auto` selects CUDA when it is available and otherwise uses the CPU.
Requesting `--device cuda` on a machine without CUDA produces an explicit
error.

### Run multiple seeds

Linux or macOS:

```bash
for seed in 12 22 32 42 52; do
  python tools/run_experiment.py \
    --config configs/bnb_multiseed.yaml \
    --seed "$seed" \
    --output-dir "outputs/BNB_seed${seed}"
done
```

Windows PowerShell:

```powershell
12,22,32,42,52 | ForEach-Object {
  python tools/run_experiment.py `
    --config configs/bnb_multiseed.yaml `
    --seed $_ `
    --output-dir "outputs/BNB_seed$_"
}
```

## Output files

Each experiment directory contains:

| File | Contents |
|---|---|
| `best_model.pt` | best validation checkpoint, configuration, and seed |
| `training_history.csv` | training loss, validation loss, and learning rate by epoch |
| `predictions_h1.csv` | single-step targets and predictions with timestamps |
| `predictions_h5.csv` | five-step recursive targets and predictions |
| `predictions_h10.csv` | ten-step recursive targets and predictions |
| `predictions_h20.csv` | twenty-step recursive targets and predictions |
| `summary.json` | seed, device, training summary, MAE, RMSE, and R2 |

Prediction files also record the forecast origin, target timestamp, target
index, and horizon step.

## Configuration reference

All experiment settings are controlled through YAML. The main keys are listed
below.

### `experiment`

| Parameter | Description |
|---|---|
| `method` | method name stored with the experiment |
| `dataset` | dataset identifier used in output naming |
| `seed` | default random seed for a single run |
| `seeds` | seed list used when launching repeated runs externally |
| `output_dir` | default output root used by `run_experiment.py` |

### `data`

| Parameter | Description |
|---|---|
| `path` | input RV CSV |
| `target_column` | RV column to forecast |
| `timestamp_column` | UTC timestamp column |
| `train_ratio` | chronological training fraction |
| `val_ratio` | chronological validation fraction |
| `anchor_lookback` | common historical anchor length |
| `short_lookback` | short-term input length |
| `long_lookback` | long-term input length |
| `tasks` | evaluated horizons, e.g. `[1, 5, 10, 20]` |
| `task_stride_equals_horizon` | use non-overlapping target blocks when `true` |

The `feature_engineering` block controls optional short-term input features. All
forecasting-performance experiments in the paper use univariate RV input, so
feature engineering is disabled in the main experiment configurations. The
multivariate setting is provided separately in
`configs/tv_multivariate_visualization.yaml` and is used only to explain the TV
feature-selection rule.

| Parameter | Paper setting | Description |
|---|---:|---|
| `enabled` | `false` | keep the forecasting input univariate when disabled |
| `hema_spans` | `[60, 720, 1440]` | HEMA windows in minute observations |
| `include_difference` | `true` | append the one-step RV difference |
| `difference_name` | `rv_difference` | name of the difference feature |
| `warmup` | `1440` | initial observations removed before splitting |

When the visualization-only configuration enables feature engineering, the
short-term feature order is fixed as:

```text
[RV, HEMA-1h, HEMA-12h, HEMA-1d, RV-difference]
```

### `model`

| Parameter | Description |
|---|---|
| `embed_dim` | Transformer embedding dimension |
| `dense_dim` | feed-forward hidden dimension |
| `num_heads` | number of attention heads |
| `dropout` | dropout probability |
| `num_blocks` | number of encoder and decoder blocks |
| `positional_encoding` | enable or disable positional encoding |
| `use_mask` | enable or disable the decoder mask |
| `max_len` | maximum positional-encoding length |

The `temporal_vector` block contains:

| Parameter | Paper setting | Description |
|---|---:|---|
| `input_feature_count` | `1` | number of short-term input features |
| `linear_feature_indices` | `[0]` | ordered features selected by the linear TV branch |
| `periodic_feature_indices` | `[0]` | ordered features selected by the periodic TV branch |

Both branches accept ordered integer index sequences. Indices must lie in
`[0, input_feature_count - 1]` and must be unique within each sequence;
overlap between branches is allowed. Selection order is preserved and remains
fixed during forward computation. Empty sequences are supported.

For input `[B, p, d]`, linear weights and biases have shape `[p, d_l]`,
and periodic weights and biases have shape `[p, d_p]`. All four tensors are
initialized independently from `N(0, 1)` and broadcast over the batch.
The branches compute `W_linear * X_l + b_linear` and
`sin(W_periodic * X_p + b_periodic)` element-wise after `index_select(2, ...)`.
Even a single-feature branch retains shape `[B, p, 1]`. Concatenation keeps
all original features first, followed by the linear and periodic outputs,
for a final shape of `[B, p, d + d_l + d_p]`. The downstream embedding derives
its input width automatically from this count.

Migration: replace `linear_feature_index: 0` with
`linear_feature_indices: [0]` and remove `init_mean` / `init_std` from older
configurations; initialization is now fixed to `N(0, 1)`. Old checkpoints
store linear weights and biases as `[p]`, whereas this implementation uses
`[p, 1]` for a one-feature branch. To reuse such a model state, explicitly
unsqueeze the final dimension of `std.temporal_vector.weights_linear` and
`std.temporal_vector.bias_linear` before loading, and retain the original
feature selection. Checkpoints do not store the index sequences, so always
reuse the matching configuration. Retrain when changing the selected features.

Thus, the TV output in the forecasting experiments has three channels: the
original RV, one learned linear feature, and one learned periodic feature. In
`configs/tv_multivariate_visualization.yaml`, `input_feature_count` is `5` and
`periodic_feature_indices` is `[0,1,2,3]`; this produces ten output channels for
the interpretability example and is not used to compute the reported
forecasting results.

The `msdcd` block contains:

| Parameter | Paper setting | Description |
|---|---:|---|
| `trend_kernel_size` | `20` | trend dynamic-convolution kernel size |
| `seasonal_kernel_size` | `7` | cyclic dynamic-convolution kernel size |
| `avg_pool_kernel_size` | `3` | initial average-pooling kernel size |
| `num_basis_kernels` | `4` | basis kernels in each dynamic layer |
| `basis_kernel_init.distribution` | `normal` | basis-kernel initialization distribution |
| `basis_kernel_init.mean` | `0.0` | normal-distribution mean |
| `basis_kernel_init.std` | `1.0` | normal-distribution standard deviation |

### `training`

| Parameter | Description |
|---|---|
| `epochs` | maximum number of training epochs |
| `batch_size` | samples per mini-batch |
| `learning_rate` | Adam learning rate |
| `num_workers` | data-loader worker processes |
| `pin_memory` | enable pinned CPU memory when CUDA is used |
| `precision` | `fp32`, `fp16`, or `bf16` |
| `gradient_clip` | maximum gradient norm; `null` disables clipping |
| `scheduler_factor` | learning-rate reduction factor |
| `scheduler_patience` | validation epochs before reducing the learning rate |
| `resource_sample_interval` | resource-monitor sampling interval in seconds |

## Configuration consistency checks

The following relationships must hold:

- `long_lookback <= anchor_lookback`;
- `short_lookback <= long_lookback`;
- `embed_dim` must be divisible by `num_heads`;
- every TV feature index must be an integer in `[0, input_feature_count - 1]`;
- neither TV index sequence may contain duplicates (cross-branch overlap is allowed);
- when feature engineering is enabled, `input_feature_count` must equal
  `2 + len(hema_spans)`;
- when feature engineering is enabled, `warmup` should be at least the largest
  HEMA span;
- the dataset must contain more rows than the warmup, anchor, and forecast
  horizon combined.

Invalid feature indices and incompatible tensor dimensions raise explicit
errors before training.

## Reproducibility

Set the random seed before constructing the model. The experiment runner does
this automatically for Python, NumPy, PyTorch CPU, and all available CUDA
devices. The configurations used for repeated experiments list the seeds
`12`, `22`, `32`, `42`, and `52`.

For strict deterministic PyTorch execution in custom scripts, use:

```python
from hdtformer.reproducibility import set_random_seed

set_random_seed(12, deterministic=True)
```

Strict deterministic execution may reduce performance and can reject CUDA
operations for which PyTorch has no deterministic implementation.

## Common problems

### `FileNotFoundError` for the dataset

Check `data.path`. Relative paths are interpreted relative to the YAML file.
With the supplied configurations, datasets belong in `Data/`.

### CUDA out-of-memory error

Reduce `training.batch_size`, or run with `--device cpu`. Mixed precision can
be enabled with `training.precision: fp16` on a compatible CUDA GPU.

### TV input-feature mismatch

Keep `model.temporal_vector.input_feature_count` consistent with the generated
features. The main forecasting configurations generate one feature, whereas
`configs/tv_multivariate_visualization.yaml` generates five.

### Positional-encoding length error

Set `model.max_len` to a value no smaller than the longest sequence presented
to the Transformer.

### Windows cannot find `python`

Use `py -3` in place of `python`, or activate the virtual environment first.
