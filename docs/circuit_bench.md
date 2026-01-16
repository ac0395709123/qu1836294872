# Circuit Optimization Benchmark Suite

This suite benchmarks the performance of various quantum circuit optimizers on a set of benchmark circuits.

## Supported Optimizers

The benchmark suite includes the following quantum circuit optimizers:

| Optimizer | Package | Description | Status |
|-----------|---------|-------------|--------|
| **Qiskit AI** | `qiskit-ibm-transpiler` | IBM AI-powered transpiler with routing and synthesis | Available |
| **GUOQ** | `wisq` / JAR | Guided Unitary Optimization for Quantum circuits | Available |
| **BQSKit** | `bqskit` | Berkeley Quantum Synthesis Toolkit (via GUOQ resynthesis) | Available |
| **TKET** | `pytket` | Quantinuum's quantum compiler with peephole optimization | Available |
| **VOQC** | `pyvoqc` | Verified Optimizer for Quantum Circuits (formally verified in Coq) | Blocked (Qiskit 1.x incompatible) |

## What ships in this repo

- **Benchmark circuits**: `benchmarks/ai_transpile/qasm/*.qasm` plus metadata describing qubit counts and baseline metrics.
- **Transpiler wrappers**: `benchmarks/ai_transpile/transpilers.py` exposes typed helpers for Qiskit AI, GUOQ, TKET, VOQC, and GUOQ via the `wisq` compiler.
- **Circuit benchmark config**: `benchmarks/ai_transpile/circuit_benchmark.yaml` lists circuits, runners, and tracked metrics.
- **Automation**: `benchmarks/ai_transpile/circuit_benchmark_runner.py` reads the YAML, executes the requested runners, and writes aggregated JSON under `reports/`.

## Prerequisites

1. Install Python deps (installs Qiskit, GUOQ requirements, etc.):
   ```bash
   uv sync
   ```

2. (Required for `tket_full_peephole` runner) Install pytket separately:
   ```bash
   uv pip install pytket
   ```
   **Note:** pytket requires networkx>=2.8.8 which conflicts with qiskit-ibm-transpiler's networkx==2.8.5 requirement. Installing pytket will upgrade networkx, which may affect some Qiskit AI transpiler features.

3. Build the GUOQ fat jar (only needed if you plan to use direct GUOQ invocation):
   ```bash
   (cd guoq && mvn package -Dmaven.test.skip)
   ```

4. (Required for WISQ runners) Set up the [wisq compiler](wisq_setup.md) so GUOQ can be invoked through wisq. This ensures Java 21 is available and the `wisq` CLI is installed. WISQ automatically manages the resynthesis server lifecycle, so no manual server startup is needed.

6. (For `voqc_nam` runner - currently blocked) VOQC requires OCaml and manual installation:
   
   **Note:** The `pyvoqc` package is currently incompatible with Qiskit 1.x due to deprecated API usage (`NoiseAdaptiveLayout`). The VOQC runner is disabled until pyvoqc is updated. Track progress at [inQWIRE/pyvoqc](https://github.com/inQWIRE/pyvoqc).
   
   When pyvoqc is updated for Qiskit 1.x compatibility:
   ```bash
   # Install opam if not already installed (see https://opam.ocaml.org/doc/Install.html)
   opam init
   eval $(opam env --switch=default)
   
   # Install the VOQC OCaml library
   opam pin voqc https://github.com/inQWIRE/mlvoqc.git#mapping
   
   # Install pyvoqc
   uv pip install git+https://github.com/inQWIRE/pyvoqc.git
   ```

## Running the suite

### Quick smoke (Qiskit AI only)

This skips all GUOQ/WISQ runners, executes only the AI transpiler variants, and drops results into `reports/circuit_benchmark/latest_results.json`.

```bash
uv run python benchmarks/ai_transpile/circuit_benchmark_runner.py \
  --skip-runner wisq_rules_only \
  --skip-runner wisq_bqskit
```

### Full comparison

Once Java 21 is configured and the `wisq` package is installed (see [wisq_setup.md](wisq_setup.md)), run all optimizers:

```bash
uv run python benchmarks/ai_transpile/circuit_benchmark_runner.py
```

Artifacts:
- Aggregated metrics: `reports/circuit_benchmark/latest_results.json`
- WISQ outputs: `reports/circuit_benchmark/wisq_rules_only/` and `reports/circuit_benchmark/wisq_bqskit/`
- TKET outputs: `reports/circuit_benchmark/tket/<runner_name>/*.qasm`
- VOQC outputs: `reports/circuit_benchmark/voqc/<runner_name>/*.qasm`

Use `--skip-runner <name>` multiple times to disable only specific runners (e.g., keep rules-only WISQ but skip the resynthesis variant).

### Running only specific optimizers

To run only TKET and VOQC (skipping WISQ):

```bash
uv run python benchmarks/ai_transpile/circuit_benchmark_runner.py \
  --skip-runner wisq_rules_only \
  --skip-runner wisq_bqskit
```

To run without VOQC (if OCaml is not installed):

```bash
uv run python benchmarks/ai_transpile/circuit_benchmark_runner.py \
  --skip-runner voqc_nam
```

### Comparing Circuit Outputs

To verify that optimized circuits are functionally equivalent to the baseline:

```bash
# Compare all runners against qiskit_ai baseline (adds comparisons to JSON)
uv run python benchmarks/ai_transpile/circuit_benchmark_runner.py \
  --compare-against qiskit_ai
```

Or compare circuits manually after running:

```bash
# Compare all circuits from results JSON
uv run python scripts/compare_circuits.py \
  reports/circuit_benchmark/latest_results.json

# Compare specific circuit
uv run python scripts/compare_circuits.py \
  reports/circuit_benchmark/latest_results.json \
  --circuit qft_8

# Compare two circuit files directly
uv run python scripts/compare_circuits.py \
  path/to/circuit1.qasm \
  path/to/circuit2.qasm
```

The comparison uses multiple methods (automatically selected):
- **QCEC** (MQT) - formal equivalence checking for larger circuits
- **Operator comparison** - for medium circuits (~12 qubits)
- **Statevector comparison** - for smaller circuits (~20 qubits)

### Pytest hook

CI-safe regression (skips WISQ optimizers that require Java) is wired through:
```bash
uv run pytest tests/test_ai_transpile_suite.py
```

## Customising the circuit benchmark config

Edit `benchmarks/ai_transpile/circuit_benchmark.yaml` to:

- Add/remove circuits by referencing new QASM paths (export more via `scripts/export_ai_transpile_circuits.py`).
- Adjust Qiskit AI optimization levels or iteration counts.
- Control WISQ objectives, gate sets, and timeouts (see [wisq_setup.md](wisq_setup.md) for parameter details).
- Toggle resynthesis: set `approx_epsilon: 0` for rules-only mode, or use a small positive value (e.g., `1e-10`) to enable BQSKit resynthesis.

Each WISQ runner accepts configuration overrides directly in the YAML.

## Troubleshooting

- **`ModuleNotFoundError: benchmarks`**: ensure you run commands from the repo root so the package path is discoverable.
- **`wisq` runner fails immediately**: confirm Java 21 is active (`$JAVA_HOME/bin/java --version` should report 21+) and the `wisq` package is installed. See [wisq_setup.md](wisq_setup.md) for a step-by-step guide.
- **`wisq` import error**: install wisq with `uv pip install wisq` or run `uv sync` to install all dependencies.
- **WISQ Java errors**: WISQ automatically manages the resynthesis server. If you encounter Java-related errors, ensure `$JAVA_HOME` points to Java 21+ and that the GUOQ jar exists in the wisq package (see [wisq_setup.md](wisq_setup.md) for details on replacing the bundled jar).
- **`pytket` not found**: install pytket with `uv pip install pytket`, or skip the TKET runner with `--skip-runner tket_full_peephole`.
- **`pyvoqc` import error**: pyvoqc is currently incompatible with Qiskit 1.x and is not installed by default. The VOQC runner is disabled. Track compatibility updates at [inQWIRE/pyvoqc](https://github.com/inQWIRE/pyvoqc).
- **VOQC unsupported gates**: VOQC doesn't support `sx` or `rxx` gates. Use circuits with compatible gate sets (h, cx, rz, t, tdg, etc.).
- **Long runtimes**: lower `iterations_per_level`, drop circuits, or use the rules-only WISQ configuration (`wisq_rules_only`) for faster optimization.
- **Comparison failures**: ensure circuits have the same number of qubits and are not too large for the selected comparison method. QCEC works best for formal equivalence checking.
