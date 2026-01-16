# Using GUOQ via wisq

The `wisq` compiler wraps GUOQ with automatic gate-set conversion, routing, and
resynthesis server management. This is the **recommended** way to use GUOQ for
benchmarking, as it eliminates the need to manually start/stop the resynthesis
server and simplifies configuration.

Follow the steps below to install Java 21, configure `wisq`, and run GUOQ
through the WISQ runners.

## 1. Install Java 21 locally

`wisq` and GUOQ both require Java 21 or newer. The helper script downloads a
Temurin build and unpacks it under `~/.local/share/java`:

```bash
chmod +x scripts/install_jdk21.sh
bash scripts/install_jdk21.sh
```

After installation, point your shell at the new JDK (add the exports to your
shell profile so future sessions inherit them):

```bash
export JAVA_HOME="$HOME/.local/share/java/jdk-21.0.5+11"
export PATH="$JAVA_HOME/bin:$PATH"
java --version  # should now report version 21
```

If you already manage Java with another tool, skip the script and ensure
`java --version` reports 21+ before installing Python packages.

## 2. Install Python dependencies (includes `wisq`)

With Java 21 on `PATH`, install the project dependencies. This pulls the `wisq`
package plus its transitive requirements:

```bash
uv sync
```

You can also install `wisq` directly inside the existing environment:

```bash
uv pip install wisq
```

## 3. Smoke-test wisq

Use the built-in example circuit to confirm `wisq` works end-to-end:

```bash
uv run wisq wisq-circuits/3_17_13.qasm -ap 1e-10 -ot 10
```

### Testing Rules-Only Mode

Run GUOQ with rewrite rules only (no resynthesis, faster):

```bash
uv run wisq benchmarks/ai_transpile/qasm/qft_8.qasm \
  --mode opt \
  --target_gateset IBMN \
  --optimization_objective TWO_Q \
  --opt_timeout 180 \
  --approx_epsilon 0 \
  --output_path reports/test/qft_8_rules_only.qasm
```

### Testing Resynthesis Mode

Run GUOQ with BQSKit resynthesis enabled (more powerful, slower):

```bash
uv run wisq benchmarks/ai_transpile/qasm/qft_8.qasm \
  --mode opt \
  --target_gateset IBMN \
  --optimization_objective TWO_Q \
  --opt_timeout 900 \
  --approx_epsilon 1e-10 \
  --output_path reports/test/qft_8_with_resynth.qasm
```

**Key Parameter: `--approx_epsilon`**
- `0` = Rules-only mode (no resynthesis server needed, faster)
- `1e-10` = Enable BQSKit resynthesis (automatically managed by WISQ, more powerful)

WISQ automatically starts and stops the resynthesis server when needed.

## 4. Running WISQ Benchmark Runners

The circuit benchmark config (`benchmarks/ai_transpile/circuit_benchmark.yaml`)
includes two WISQ runners that replace the direct GUOQ invocation:

- **`wisq_rules_only`**: Rewrite rules only (`approx_epsilon: 0`), fast optimization
- **`wisq_bqskit`**: With BQSKit resynthesis (`approx_epsilon: 1e-10`), powerful optimization

### Run all optimizers (including both WISQ runners)

```bash
uv run python benchmarks/ai_transpile/circuit_benchmark_runner.py
```

### Run only rules-only optimization

```bash
uv run python benchmarks/ai_transpile/circuit_benchmark_runner.py \
  --skip-runner wisq_bqskit
```

### Run only with resynthesis

```bash
uv run python benchmarks/ai_transpile/circuit_benchmark_runner.py \
  --skip-runner wisq_rules_only
```

### Skip all WISQ runners

```bash
uv run python benchmarks/ai_transpile/circuit_benchmark_runner.py \
  --skip-runner wisq_rules_only \
  --skip-runner wisq_bqskit
```

Artifacts land under:
- `reports/circuit_benchmark/wisq_rules_only/` - Rules-only optimization results
- `reports/circuit_benchmark/wisq_bqskit/` - Resynthesis-enabled optimization results

## 5. Resource Limiting for Remote Servers

WISQ/GUOQ resynthesis optimization is computationally intensive and can consume
all available CPU cores and memory. On remote servers, this can make the system
completely unresponsive, preventing SSH reconnection. To prevent this, use the
resource-limiting wrapper scripts.

### Default Resource Limiting (Recommended)

The `run_wisq_safe.sh` wrapper limits BQSKit workers to 50% of CPU cores and
runs with lower priority:

```bash
./scripts/run_wisq_safe.sh uv run wisq circuit.qasm -ap 1e-10 -ot 300
```

The benchmark tmux script automatically uses this wrapper, so running benchmarks
via tmux is safe by default:

```bash
./scripts/run_circuit_benchmark_tmux.sh create
```

### Configuration via Environment Variables

Customize resource limits using environment variables:

```bash
# Use only 4 BQSKit workers (explicit count)
BQSKIT_NUM_WORKERS=4 ./scripts/run_wisq_safe.sh uv run wisq circuit.qasm

# Use 25% of available CPU cores
BQSKIT_WORKER_FRACTION=0.25 ./scripts/run_wisq_safe.sh uv run wisq circuit.qasm

# Adjust CPU priority (higher nice = lower priority, range 0-19)
WISQ_NICE_LEVEL=10 ./scripts/run_wisq_safe.sh uv run wisq circuit.qasm
```

### Strict Resource Limits with systemd

For hard resource limits that cannot be exceeded (recommended for shared servers
or when SSH responsiveness is critical), use `run_wisq_systemd.sh`:

```bash
# Default: 2 CPU cores worth, 8GB RAM, no swap
./scripts/run_wisq_systemd.sh uv run wisq circuit.qasm -ap 1e-10 -ot 300

# Custom limits: 4 cores, 16GB RAM
WISQ_CPU_QUOTA=400 WISQ_MEMORY_MAX=16G ./scripts/run_wisq_systemd.sh uv run wisq circuit.qasm

# Very conservative (1 core, 4GB RAM)
WISQ_CPU_QUOTA=100 WISQ_MEMORY_MAX=4G ./scripts/run_wisq_systemd.sh uv run wisq circuit.qasm
```

**Note:** `CPU_QUOTA` is percentage, not core count: `100%` = 1 core, `400%` = 4 cores.

Setting `WISQ_MEMORY_SWAP_MAX=0` (the default) prevents swap thrashing, which can
freeze systems even when memory isn't fully exhausted.

### YAML Configuration

You can also configure worker limits in `circuit_benchmark.yaml`:

```yaml
runners:
  - name: wisq_bqskit
    type: wisq
    target_gateset: IBMN
    optimization_objective: TWO_Q
    approx_epsilon: 1e-10
    opt_timeout: 900
    bqskit_num_workers: 4        # Explicit worker count
    # OR
    bqskit_worker_fraction: 0.25 # Fraction of CPU cores
```

### Troubleshooting

If the system becomes unresponsive during WISQ optimization:

1. **Prevention**: Always use `run_wisq_safe.sh` or `run_wisq_systemd.sh` wrappers
2. **If stuck**: Wait for the optimization timeout (`opt_timeout`) to expire
3. **Hard reboot**: If SSH is completely unresponsive, you may need physical access
   or out-of-band management (IPMI/iDRAC) to reboot

For long-running benchmarks, the tmux script is recommended as it:
- Automatically applies resource limits
- Persists across SSH disconnections
- Logs output for later review

Refer back to [`docs/circuit_bench.md`](circuit_bench.md) for the complete workflow
(exporting circuits, comparing optimizers, and interpreting the reports).
