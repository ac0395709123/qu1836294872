# PyTKET Sub-Environment Setup

## Overview

This project uses a **dual-environment setup** to handle incompatible dependency requirements:

- **Main environment** (`.venv`): Contains all project dependencies including `qiskit-ibm-ai-local-transpiler`
- **TKET environment** (`.venv-tket`): Isolated environment for PyTKET and its dependencies

## Why is this necessary?

PyTKET and the IBM AI Local Transpiler have conflicting networkx requirements:

- `pytket>=1.36.1` requires `networkx>=2.8.8`
- `qiskit-ibm-ai-local-transpiler` requires `networkx==2.8.5`

While both packages *import* successfully with networkx 3.x, the AI transpiler **breaks at runtime** with newer networkx versions, producing errors like:

```
WARNING:qiskit_ibm_transpiler.wrappers.ai_local_synthesis:ERROR. No model available for the requested subgraph
```

## Setup

### Automatic Setup

Run the setup script:

```bash
./scripts/setup_tket_env.sh
```

### Manual Setup

```bash
# Create TKET virtual environment
uv venv .venv-tket --python 3.12

# Install PyTKET and dependencies
uv pip install --python .venv-tket/bin/python pytket qiskit numpy
```

## Usage

### Method 1: Using the tket_runner Module (Recommended)

The `benchmarks.tket_runner` module provides utilities to run PyTKET code in the isolated environment:

```python
from benchmarks.tket_runner import run_tket_script

# Run a PyTKET script
script = """
from pytket import Circuit

circuit = Circuit(3)
circuit.H(0)
circuit.CX(0, 1)
print(f"Gates: {circuit.n_gates}")
"""

result = run_tket_script(script)
print(result.stdout)
```

### Method 2: Direct Subprocess Invocation

For standalone scripts:

```python
import subprocess
from benchmarks.tket_runner import get_tket_python_path

tket_python = get_tket_python_path()
subprocess.run([str(tket_python), "my_tket_script.py"])
```

### Method 3: Direct Python Interpreter

For interactive work or testing:

```bash
# Using the convenience wrapper (recommended)
./scripts/tket my_tket_script.py           # Run a script
./scripts/tket -c "import pytket; ..."     # Run a command
./scripts/tket                             # Interactive shell

# Or use the Python interpreter directly
.venv-tket/bin/python my_tket_script.py

# Start an interactive session
.venv-tket/bin/python
>>> from pytket import Circuit
>>> # ... your TKET code here
```

## Usage Examples

The `benchmarks.tket_runner` module provides the interface for running TKET code. See the test files in `tests/test_tket_runner.py` for working examples demonstrating:

1. Simple PyTKET circuit creation
2. Qiskit → TKET conversion and optimization
3. Direct subprocess invocation

## Verification

Check both environments:

```python
from benchmarks.tket_runner import print_environment_info

print_environment_info()
```

Expected output:

```
=== Main Environment ===
networkx: 2.8.5
pytket: not installed

=== TKET Environment ===
networkx: 3.6.1
pytket: 2.11.0
```

## Maintenance

### Updating PyTKET

```bash
uv pip install --python .venv-tket/bin/python --upgrade pytket
```

### Adding More Packages to TKET Environment

```bash
uv pip install --python .venv-tket/bin/python <package-name>
```

### Recreating the Environment

```bash
rm -rf .venv-tket
./scripts/setup_tket_env.sh
```

## Architecture

```
quantum-transpiler-benchmark/
├── .venv/                      # Main environment (networkx 2.8.5)
│   └── qiskit-ibm-ai-local-transpiler, qiskit, ...
├── .venv-tket/                 # TKET environment (networkx 3.6.1)
│   └── pytket, qiskit, numpy
├── benchmarks/
│   └── tket_runner.py          # Utilities for TKET sub-environment
├── tests/
│   └── test_tket_runner.py     # Usage examples and tests
└── scripts/
    └── setup_tket_env.sh       # Setup script
```

## Troubleshooting

### "TKET environment not found" error

Run the setup script:

```bash
./scripts/setup_tket_env.sh
```

### Import errors in TKET environment

Make sure all required packages are installed in the TKET environment:

```bash
uv pip install --python .venv-tket/bin/python pytket qiskit numpy
```

### Serialization errors with run_tket_function()

The `run_tket_function()` utility requires JSON-serializable arguments. For complex objects:

1. Convert to QASM/JSON in main environment
2. Pass as string
3. Deserialize in TKET environment

See `tests/test_tket_runner.py` for patterns and examples.
