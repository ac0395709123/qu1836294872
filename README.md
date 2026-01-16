# Quantum Circuit Transpiler Benchmark Suite

A comprehensive benchmark suite for comparing quantum circuit optimization methods and exploring **optimization chaining**: treating circuit optimizers as composable transformations and learning sequences that improve circuit metrics under compilation-time budgets.

## Overview

This repository provides infrastructure for benchmarking multiple quantum circuit optimizers on NISQ-scale benchmark circuits. The suite enables systematic comparison of optimization quality, runtime performance, and circuit equivalence verification across different transpilation approaches.

Recent advances have produced diverse optimization techniques—from heuristic rewriting to AI-powered synthesis—but no single optimizer dominates across all circuits and hardware. Moreover, modern optimizers like IBM's AI transpiler are stochastic, creating both opportunity (some runs are much better) and challenge (unclear when to retry or switch tools).

This work explores **optimization chaining**: composing heterogeneous optimizers into sequences that can outperform any single optimizer. We provide:
1. A benchmark harness for head-to-head optimizer evaluation
2. Infrastructure for chaining optimizers and measuring synergies
3. An RL-based orchestrator (in development) that learns effective optimization sequences under time budgets

### Research Questions

**RQ1: AI Transpiler Variability**  
How much variability exists in IBM's qiskit-ibm-transpiler across optimization levels and repeated runs?

**RQ2: Optimizer Improvements**  
Given a circuit produced by standard Qiskit transpiler baselines, how much additional improvement in two-qubit gate count, depth, and T-count is available by applying state-of-the-art optimizers such as AI-powered transpilation, GUOQ-style rewrite+resynthesis, and RL-based synthesis/routing?

**RQ3: RL-Based Orchestration**  
Can we develop a reinforcement learning (RL) agent to learn effective sequences of optimizations under a time budget?

**RQ4: Optimizer Synergies**  
Which combinations and orderings of optimizers are synergistic or antagonistic, and when does chaining beat the best single technique?

## Supported Optimizers

| Optimizer | Package | Description | Status |
|-----------|---------|-------------|--------|
| **Qiskit AI** | `qiskit-ibm-transpiler` | IBM AI-powered transpiler with routing and synthesis | Available |
| **GUOQ** | `wisq` / JAR | Guided Unitary Optimization for Quantum circuits | Available |
| **BQSKit** | `bqskit` | Berkeley Quantum Synthesis Toolkit (via GUOQ resynthesis) | Available |
| **TKET** | `pytket` | Quantinuum's quantum compiler with peephole optimization | Available |
| **Qiskit Standard** | `qiskit` | Standard Qiskit transpiler with Sabre routing | Available (baseline) |

## Installation

### Prerequisites

- **Python 3.12** (required for dependency compatibility)
- **uv** package manager ([installation instructions](https://github.com/astral-sh/uv))
- **Java 21** (required for GUOQ/WISQ optimizers)
- **Git** with submodule support

### Quick Start

1. **Clone the repository with submodules:**

```bash
git clone --recursive <repository-url>
cd quantum-transpiler-benchmark
```

If you already cloned without `--recursive`, initialize submodules:

```bash
git submodule update --init --recursive
```

2. **Install Python dependencies:**

```bash
uv sync
```

This installs Qiskit, IBM AI transpiler, BQSKit, and other core dependencies.

### Optional: Java 21 Setup for GUOQ/WISQ

To use GUOQ and WISQ optimizers, you need Java 21 or newer:

```bash
# Use the provided installer script
chmod +x scripts/install_jdk21.sh
bash scripts/install_jdk21.sh

# Configure your shell (add to ~/.bashrc or ~/.zshrc)
export JAVA_HOME="$HOME/.local/share/java/jdk-21.0.5+11"
export PATH="$JAVA_HOME/bin:$PATH"

# Verify installation
java --version  # should show version 21+
```

See [`docs/wisq_setup.md`](docs/wisq_setup.md) for detailed WISQ configuration and troubleshooting.

### Optional: TKET Environment

PyTKET has conflicting dependencies with the IBM AI transpiler (networkx versions). To use TKET:

```bash
# Create isolated TKET environment
./scripts/setup_tket_env.sh

# TKET is now available via the tket_runner module
# See docs/tket_environment.md for usage details
```

## Running Benchmarks

### Quick Benchmark (Qiskit AI only)

Run a fast benchmark using only the AI transpiler:

```bash
uv run python benchmarks/ai_transpile/circuit_benchmark_runner.py \
  --skip-runner wisq_rules_only \
  --skip-runner wisq_bqskit
```

Results are saved to `reports/circuit_benchmark/latest_results.json`.

### Full Benchmark Suite

Run all optimizers (requires Java 21 setup):

```bash
uv run python benchmarks/ai_transpile/circuit_benchmark_runner.py
```

This benchmarks:
- Qiskit AI transpiler (optimization levels 1, 2, 3 with 3 iterations each)
- Standard Qiskit transpiler (optimization levels 1, 2, 3)
- WISQ rules-only optimization
- WISQ with BQSKit resynthesis
- TKET full peephole optimization

Artifacts are saved under `reports/circuit_benchmark/`:
- `latest_results.json` - Aggregated metrics
- `wisq_rules_only/` - WISQ optimized circuits (rules only)
- `wisq_bqskit/` - WISQ optimized circuits (with resynthesis)
- `tket/` - TKET optimized circuits

### Benchmark Configuration

Edit [`benchmarks/ai_transpile/circuit_benchmark.yaml`](benchmarks/ai_transpile/circuit_benchmark.yaml) to:
- Add/remove benchmark circuits
- Configure optimization parameters
- Enable/disable specific runners
- Adjust timeouts and worker counts

See [`docs/circuit_bench.md`](docs/circuit_bench.md) for complete documentation.

## Preliminary Results

Our experiments on representative NISQ-scale benchmarks (QFT, EfficientSU2, RealAmplitudes) provide evidence for optimization chaining:

**RQ1: AI Transpiler Variability**
- Measurable variability across optimization levels and repeated runs
- Coefficient of variation (CV) grows with optimization level: 0.31 (level 1) → 0.46 (level 3)
- Higher levels can regress on some circuits (e.g., EfficientSU2₁₂ at level 3)

**RQ2: Headroom Beyond Baselines**
- AI transpiler: 54% reduction in two-qubit gates (QFT₈), 53% (EfficientSU2₁₂) vs. standard Qiskit
- TKET: 51-73% reduction in two-qubit gates across benchmarks
- WISQ: 59-96% reduction, with BQSKit resynthesis providing additional gains

**RQ4: Evidence of Synergy**
- Simple two-stage chains outperform best single optimizer:
  - QFT₈: TKET → WISQ reduces two-qubit gates from 43 (best single) to 35 (chain)
  - EfficientSU2₁₂: WISQ → TKET matches best gate count (11) while improving depth 38 → 30
- Chains can improve secondary metrics without sacrificing primary ones
- Faster chains (e.g., Qiskit → WISQ) match quality while reducing runtime

These results suggest that optimizer composition is feasible and promising for scalable quantum compilation.

## Optimization Chaining

### Chain Experiments (RQ4)

Test whether short optimizer chains can outperform the best single optimizer:

```bash
# Run chain experiments (customizable in benchmarks/ai_transpile/chain_experiment.yaml)
uv run python benchmarks/ai_transpile/chain_executor.py \
  --config benchmarks/ai_transpile/chain_experiment.yaml \
  --output-dir reports/chain_experiment
```

Example chains:
- **TKET → WISQ**: Apply TKET peephole optimization, then WISQ rules
- **Qiskit → TKET**: Standard transpilation followed by peephole optimization
- **WISQ+BQSKit → TKET**: Resynthesis followed by local optimization

Results show that two-stage chains can improve two-qubit gate count and depth beyond the best single optimizer on circuits like QFT and EfficientSU2.

### RL-Based Orchestrator (RQ3 - In Development)

The RL orchestrator models compilation as a Markov Decision Process (MDP) where:
- **States**: Circuit representations (depth, gate counts, remaining time budget, optional approximation budget)
- **Actions**: Selecting an optimizer and its parameters (optimization level, retry count, approximation tolerance)
- **Transitions**: Deterministic given tool output, but potentially stochastic if optimizer has randomness
- **Rewards**: Shaped to encourage metric reductions while penalizing runtime and approximation error:
  ```
  R(s,a,s') = ΔM - λ·t(a) - μ·ε(a)
  ```
  where ΔM is the metric improvement, t(a) is runtime cost, and ε(a) is approximation parameter

**Correctness guarantees:**
- Equivalence checking validates semantic preservation after each transformation
- Controlled-error compilation tracks error accumulation for approximate optimizers
- Budget violations treated as terminal failures with negative reward

**Training strategies:**
- Off-policy RL (DQN) or policy-gradient methods (PPO)
- Bootstrapping from logged optimization trajectories
- Imitation learning or bandit-style exploration before full RL training

See `benchmarks/ai_transpile/rl_orchestrator.py` for the current implementation.

## Analysis and Figures

### RQ1: AI Transpiler Variability Analysis

Analyze variability across AI transpiler optimization levels and seeds:

```bash
uv run python scripts/analyze_rq1_variability.py \
  --results reports/circuit_benchmark/latest_results.json \
  --output-dir reports/paper_figures/rq1
```

Generates:
- Coefficient of variation plots
- Variance boxplots
- Statistical summary tables

### RQ2: Optimizer Improvement Analysis

Compare optimizer improvements over baselines:

```bash
uv run python scripts/analyze_rq2_improvements.py \
  --results reports/circuit_benchmark/latest_results.json \
  --output-dir reports/paper_figures/rq2
```

Generates:
- Improvement bar charts
- Runtime vs. improvement scatter plots
- Comparative statistics

### Generate All Paper Figures

Run all analyses at once:

```bash
uv run python scripts/generate_paper_figures.py \
  --results reports/circuit_benchmark/latest_results.json \
  --output-dir reports/paper_figures
```

## Repository Structure

```
quantum-transpiler-benchmark/
├── benchmarks/
│   ├── tket_runner.py           # TKET isolated environment runner
│   └── ai_transpile/            # Core benchmark infrastructure
│       ├── circuit_benchmark_runner.py  # Main benchmark orchestrator
│       ├── circuit_benchmark.yaml       # Benchmark configuration
│       ├── chain_executor.py            # Optimizer chaining experiments
│       ├── chain_experiment.yaml        # Chain configuration
│       ├── rl_orchestrator.py           # RL-based optimization orchestrator
│       ├── transpilers.py               # Optimizer wrapper interfaces
│       ├── statistics.py                # Statistical analysis utilities
│       ├── visualization.py             # Plotting utilities
│       ├── circuit_comparison.py        # Equivalence checking
│       └── qasm/                        # Benchmark circuits
├── scripts/
│   ├── analyze_rq1_variability.py       # RQ1 analysis
│   ├── analyze_rq2_improvements.py      # RQ2 analysis
│   ├── generate_paper_figures.py        # Master figure generator
│   ├── compare_circuits.py              # Circuit equivalence checker
│   ├── install_jdk21.sh                 # Java 21 installer
│   ├── setup_tket_env.sh                # TKET environment setup
│   └── run_wisq_safe.sh                 # Resource-limited WISQ wrapper
├── docs/
│   ├── circuit_bench.md         # Benchmark workflow guide
│   ├── wisq_setup.md            # GUOQ/WISQ setup guide
│   └── tket_environment.md      # TKET dual-environment guide
├── tests/                       # Unit and integration tests
│   ├── conftest.py              # Pytest configuration
│   ├── test_circuit_benchmark_runner.py
│   ├── test_chain_executor.py
│   ├── test_transpilers.py
│   └── ...                      # Additional test modules
├── reports/                     # Benchmark outputs (gitignored)
│   ├── circuit_benchmark/       # Circuit benchmark results
│   └── paper_figures/           # Publication-ready figures
├── pyproject.toml               # Python dependencies
├── uv.lock                      # Locked dependency versions
└── LICENSE                      # MIT License
```

## Circuit Equivalence Verification

Verify that optimized circuits are functionally equivalent to baselines:

```bash
# Compare all circuits in results JSON
uv run python scripts/compare_circuits.py \
  reports/circuit_benchmark/latest_results.json

# Compare specific circuit
uv run python scripts/compare_circuits.py \
  reports/circuit_benchmark/latest_results.json \
  --circuit qft_8

# Compare two QASM files directly
uv run python scripts/compare_circuits.py \
  path/to/circuit1.qasm \
  path/to/circuit2.qasm
```

The comparison tool automatically selects appropriate methods:
- **QCEC** (MQT) - Formal equivalence checking for larger circuits
- **Operator comparison** - For medium circuits (~12 qubits)
- **Statevector comparison** - For smaller circuits (~20 qubits)

## Testing

Run the test suite:

```bash
# All tests
uv run pytest tests/

# Specific test file
uv run pytest tests/test_circuit_benchmark_runner.py

# With coverage
uv run pytest --cov=benchmarks --cov-report=html
```

## Linting and Type Checking

```bash
# Lint code
uv run ruff check

# Type check
uv run ty
```

## Troubleshooting

### WISQ Runner Fails

**Symptom:** `wisq` runner fails immediately or with Java errors.

**Solution:**
1. Verify Java 21 is active: `java --version` should report 21+
2. Ensure `JAVA_HOME` is set correctly
3. Install wisq: `uv pip install wisq`
4. See [`docs/wisq_setup.md`](docs/wisq_setup.md) for detailed troubleshooting

### TKET Not Found

**Symptom:** `pytket` import errors or TKET runner fails.

**Solution:**
1. Run `./scripts/setup_tket_env.sh` to create isolated TKET environment
2. Or skip TKET: `--skip-runner tket_full_peephole`
3. See [`docs/tket_environment.md`](docs/tket_environment.md) for details

### Networkx Version Conflicts

**Symptom:** AI transpiler produces warnings or errors about networkx.

**Solution:**
The AI transpiler requires `networkx==2.8.5` but pytket requires `>=2.8.8`. Use the dual-environment setup:
- Main environment: AI transpiler (networkx 2.8.5)
- TKET environment: pytket (networkx 3.x)

### System Becomes Unresponsive During WISQ

**Symptom:** System freezes or SSH becomes unresponsive during resynthesis.

**Solution:**
Use resource-limiting wrappers:
```bash
# Conservative resource limits
./scripts/run_wisq_safe.sh uv run python benchmarks/ai_transpile/circuit_benchmark_runner.py

# Or use systemd-based hard limits
./scripts/run_wisq_systemd.sh uv run wisq circuit.qasm -ap 1e-10 -ot 300
```

See [`docs/wisq_setup.md`](docs/wisq_setup.md) section 5 for detailed resource management.

## Related Publication

This benchmark suite accompanies the research paper:

**"Orchestrating Quantum Circuit Optimization"**

The paper proposes optimization chaining as a research direction for quantum compilation, arguing that treating optimizers as composable transformations and learning effective sequences can outperform fixed one-shot optimization strategies.

Key contributions:
- Formalization of optimization chaining as sequential decision-making
- Empirical evidence of AI transpiler variability and optimizer headroom
- Demonstration that short chains can beat the best single optimizer
- RL-based orchestrator framework for learning optimization sequences

## Citation

```bibtex
[Citation information will be added upon publication]
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2025 Anonymous
