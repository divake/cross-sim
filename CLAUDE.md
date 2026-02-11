# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CrossSim (V3.1.1) is a GPU-accelerated, Python-based crossbar simulator for analog in-memory computing. It provides a Numpy-like API (`AnalogCore`) that can be used as a drop-in replacement for Numpy matrix operations to emulate deployment on analog hardware. It models device and circuit non-idealities (programming errors, conductance drift, read noise, ADC precision loss, parasitic resistances). Built by Sandia National Laboratories; Python 3.10/3.11.

## Common Commands

```bash
# Install
pip install .                     # Basic install
pip install --editable .          # Editable (development) install
pip install ".[dev]"              # With dev tools (pytest, ruff, mypy, etc.)
pip install ".[gpu]"              # With CuPy for GPU acceleration

# Testing
pytest                            # Run all tests (testpaths = tests/)
pytest tests/test_foo.py          # Run a single test file
pytest -k "test_name"             # Run a specific test
tox                               # Run tests across Python 3.10 and 3.11

# Linting & Formatting
ruff check simulator/             # Lint (line-length 88, Google docstrings)
mypy simulator/                   # Type checking
black simulator/                  # Format code

# Documentation
sphinx-build -b html docs/sphinx/source/ docs/sphinx/build/
sphinx-apidoc simulator/ -o docs/sphinx/source/   # Add new modules to docs
```

## Architecture

### Core Hierarchy (`simulator/cores/`)

The core abstraction models physical resistive crossbar arrays:

- **`AnalogCore`** — Top-level user-facing API. Numpy-like interface for matrix multiplication. Internally partitions matrices across multiple physical cores based on data types and parameters.
- **`WrapperCore`** — Base class for core types that wrap `NumericCore`.
  - **`BalancedCore`** — Differential encoding for negative values (two arrays).
  - **`OffsetCore`** — Offset-based negative number handling.
  - **`BitslicedCore`** — Bit-sliced precision handling (wraps BalancedCore/OffsetCore).
- **`NumericCore`** — Represents a single physical resistive crossbar. Handles device programming, MVM operations, ADC/DAC conversion, and parasitic circuit simulation.
- **`ICore`** — Abstract interface defining the core contract.

### Parameters System (`simulator/parameters/`)

Hierarchical, JSON-serializable parameter objects:

- **`CrossSimParameters`** — Top-level container; the main object users create and configure.
  - `core_parameters` — Matrix mapping, core style (balanced/offset/bitsliced), partitioning.
  - `xbar_parameters` — Device model, array dimensions, ADC/DAC settings, circuit parasitics.
  - `simulation_parameters` — GPU toggle, convergence settings, profiling.
- **`BaseParameters`** — Base class with `from_json()`/`to_json()` for all parameter objects.
- Pre-configured architectures live in `simulator/configs/` (ISAAC, PRIME, SONOS-TCAS JSON files).

### Backend Abstraction (`simulator/backend/`)

`ComputeBackend` is a singleton that abstracts Numpy vs. CuPy. All numeric code uses `xp = ComputeBackend()` to get the current array library, enabling transparent GPU acceleration.

### DNN Integration (`simulator/algorithms/dnn/`)

Neural network layers that route matrix operations through `AnalogCore`:

- **PyTorch** (`dnn/torch/`): `AnalogLinear`, `AnalogConv1d/2d/3d`. Supports inference and hardware-aware training via backpropagation.
- **Keras** (`dnn/keras/`): `AnalogDense`, `AnalogConv1D/2D/3D`, `AnalogDepthwiseConv2D`. Inference only.
- Both interfaces take existing models and replace compatible layers with analog equivalents using `from_torch()`/`from_keras()` class methods.
- The old `DNN` class (`algorithms/dnn/dnn.py`) is deprecated since 3.1, to be removed in 3.2.

### Devices & Circuits

- **`simulator/devices/`**: Device models with `IDevice` interface. `GenericDevice` for parameterizable models; `custom/` directory for specific technologies (SONOS, PCM, RRAM variants).
- **`simulator/circuits/`**: ADC models (`adc/`), DAC models (`dac/`), and array circuit simulators (`array/`) for parasitic resistance modeling across 4 topologies.

### Applications (`applications/`)

Example scripts demonstrating usage — not part of the simulator package itself. Includes DNN inference examples (PyTorch and Keras), DSP (DFT) examples, and parameter configuration templates.

## Key Conventions

- All source files carry Sandia copyright headers (2017-2024).
- Uses `from __future__ import annotations` for Python 3.10+ type hint syntax.
- Google-style docstrings (enforced by ruff `D` rules).
- `__init__.py` files are exempted from E402 and F401 lint rules.
- `algorithms/archive/` is excluded from all linting.
- Line length: 88 characters (Black-compatible).
