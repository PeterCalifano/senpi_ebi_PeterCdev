# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

SENPI (Synthetic Events Through Neural Processing and Integration) is a PyTorch-based Python package for event-based vision cameras. It simulates realistic event camera data from photometric frames and provides tools for processing, filtering, visualizing, and reconstructing intensity frames from event streams.

## Environment Setup

Two conda environment files are provided:

- `environment.yml` — Windows-oriented (references `C:\Users\jgreene97\.conda\envs\senpi`)
- `environment-linux.yml` — Linux-oriented (named `pyopt`)

```bash
conda env create --name senpi --file=environment.yml   # or environment-linux.yml on Linux
conda activate senpi
```

Key dependencies: Python 3.11, PyTorch 2.1.2 (CUDA 11.8), NumPy, Pandas, Kornia, OpenCV, Matplotlib, tqdm, einops.

## Running Notebooks

The primary usage examples are in Jupyter notebooks at the repo root:

- `Official_SENPI_Simulation_Testing_Notebook.ipynb` — simulator usage
- `Official_SENPI_Events_Processing_Testing_Notebook.ipynb` — event processing
- `Official_SENPI_Simulation_Differentiability_Testing_Notebook.ipynb` — autograd/differentiability
- `Demo_load_and_process_video.ipynb` — loading and processing video

```bash
jupyter lab
```

## Architecture

The `senpi/` package is structured as:

- **`senpi/__init__.py`** — flat re-exports everything via `import *` from all submodules. Users typically `import senpi` and call `senpi.X` directly.

- **`senpi/sim/`** — Event camera simulator
  - `params.py`: `make_params()` returns a default params dict (thresholds, noise, well capacity, refractory period, CUDA device, etc.)
  - `simulator.py`: `EventSimulator(nn.Module)` — takes photometric tensor `[F, H, W]`, outputs event tensor `[N, 4]` in `[t, x, y, p]` format
  - `simulator_utils.py`: Utility functions for the simulator

- **`senpi/data_manip/`** — Event stream manipulation (all differentiable)
  - `conversions.py`: `event_stream_to_frame_vol`, `frame_vol_to_event_stream`, time unit converters, 1D tensor/dataframe/numpy converters
  - `filters.py`: `Filter` base class → `PolarityFilter`, `BAFilter` (background activity), `IEFilter`, `YNoiseFilter`
  - `algs.py`: `Alg` base class → `FlipXAlg`, `FlipYAlg`, `InvertPolarityAlg`
  - `preprocessing.py`: Conversions between numpy structured arrays, tensors, and DataFrames; batch operations
  - `computation.py`: `SequentialCompute` — chains multiple filters/algs; `has_parameter` utility

- **`senpi/data_io/`** — I/O utilities (`basic_utils.py`): load/save event data between CSV, DataFrames, NumPy arrays, and PyTorch tensors; dtype conversion helpers

- **`senpi/data_vis/`** — Visualization (`visualization.py`): `plot_events`, `plot_time_surface`

- **`senpi/data_gen/`** — Reconstruction (`reconstruction.py`): `ImgFramesFromEventsGenerator` — implements continuous-time ODE-based intensity reconstruction (Scheerlinck et al., ACCV 2018)

- **`senpi/custom_classes/`** — `classes.py`: `CustomComparison` — a `torch.autograd.Function` that implements differentiable comparison operations (hard forward, sigmoid-based backward) to replace `torch.where` while preserving gradients

- **`senpi/constants.py`** — Camera-specific constants and dtype mappings for Prophesee (1280×720, microseconds) and DAVIS 240C (240×180, seconds); `TimeUnit` enum

## Key Concepts

**Data format**: Event streams are `(N, 4)` or `(N, 5)` tensors/DataFrames/NumPy arrays with columns `['t', 'x', 'y', 'p']` or `['b', 't', 'x', 'y', 'p']`. The `order` parameter (a list of strings) specifies column ordering for tensors.

**Binary vs signed polarity**: Polarity values are either `{0, 1}` (binary mode) or `{-1, 1}` (signed mode). Many functions accept a `binary_polarity` flag.

**Differentiability**: All core operations preserve PyTorch autograd. The `CustomComparison` class in `custom_classes/classes.py` enables gradient flow through threshold comparisons in the simulator.

**Simulator usage**:

```python
from senpi.sim.params import make_params
from senpi.sim.simulator import EventSimulator

params = make_params()
sim = EventSimulator(params)
events = sim.forward(photometric_tensor)  # photometric_tensor: [F, H, W]
```
