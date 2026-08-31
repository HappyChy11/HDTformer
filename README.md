# HDTformer
# HDTformer Source Package

This folder is the open-source code package created for uploading to GitHub.

## Structure

- `hdtformer/` core implementation (config/data/model/engine/metrics/resources/transformer/msdcd/temporal_vector).
- `configs/` method/reproducible configs (`default`, `bnb_multiseed`, `btc_multiseed`, `bnb_yearly`).
- `tools/` added helpers for visualization source data:
  - `export_msdcd_outputs.py`
  - `export_tv_outputs.py`

## Requirements

```bash
python3 -m pip install -r requirements.txt
```

Python packages are:
- numpy
- pandas
- scikit-learn
- torch
- PyYAML
- psutil
- tqdm

## Scope

For inference and visualization preparation, use:
- `tools/export_msdcd_outputs.py`
- `tools/export_tv_outputs.py`

