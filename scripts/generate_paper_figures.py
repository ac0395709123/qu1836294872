"""Master script to generate all paper figures and tables for NIER/IVR submission.

This script orchestrates RQ1 and RQ2 analyses to produce publication-ready
figures and tables suitable for inclusion in the paper.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_analysis_script(
    script_name: str,
    results_file: Path,
    output_dir: Path,
    additional_args: list[str] | None = None,
) -> bool:
    """Run an analysis script and handle errors.

    Args:
        script_name: Name of the script to run
        results_file: Path to results JSON file
        output_dir: Output directory
        additional_args: Additional command-line arguments

    Returns:
        True if successful, False otherwise
    """
    script_path = Path(__file__).parent / script_name

    if not script_path.exists():
        print(f"✗ Script not found: {script_path}")
        return False

    cmd = [
        "uv",
        "run",
        "python",
        str(script_path),
        "--results",
        str(results_file),
        "--output-dir",
        str(output_dir),
    ]

    if additional_args:
        cmd.extend(additional_args)

    print(f"\n{'='*80}")
    print(f"Running: {script_name}")
    print(f"{'='*80}")

    try:
        subprocess.run(cmd, check=True, capture_output=False, text=True)
        print(f"✓ {script_name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {script_name} failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"✗ {script_name} failed: {e}")
        return False


def main() -> None:
    """Generate all paper figures and tables."""
    parser = argparse.ArgumentParser(description="Generate all paper figures for NIER/IVR submission")
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("reports/circuit_benchmark/full/latest_results.json"),
        help="Path to benchmark results JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/paper_figures"),
        help="Output directory for figures and tables",
    )
    parser.add_argument(
        "--latex",
        action="store_true",
        help="Export tables as LaTeX",
    )
    parser.add_argument(
        "--skip-rq1",
        action="store_true",
        help="Skip RQ1 analysis",
    )
    parser.add_argument(
        "--skip-rq2",
        action="store_true",
        help="Skip RQ2 analysis",
    )
    args = parser.parse_args()

    # Validate inputs
    if not args.results.exists():
        print(f"✗ Results file not found: {args.results}")
        print("\nPlease run the benchmark suite first:")
        print("  uv run python benchmarks/ai_transpile/circuit_benchmark_runner.py")
        sys.exit(1)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("PAPER FIGURE GENERATION FOR NIER/IVR SUBMISSION")
    print("="*80)
    print(f"Results: {args.results}")
    print(f"Output: {args.output_dir}")
    print(f"LaTeX export: {args.latex}")
    print()

    success_count = 0
    total_count = 0

    # Additional arguments for scripts
    extra_args = []
    if args.latex:
        extra_args.append("--latex")

    # RQ1: Variability Analysis
    if not args.skip_rq1:
        total_count += 1
        if run_analysis_script("analyze_rq1_variability.py", args.results, args.output_dir, extra_args):
            success_count += 1

    # RQ2: Improvement Analysis
    if not args.skip_rq2:
        total_count += 1
        if run_analysis_script("analyze_rq2_improvements.py", args.results, args.output_dir, extra_args):
            success_count += 1

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Completed: {success_count}/{total_count} analyses")

    if success_count == total_count:
        print("\n✓ All figures and tables generated successfully!")
        print(f"\nOutput location: {args.output_dir.absolute()}")
        print("\nGenerated files:")

        # List generated files
        if args.output_dir.exists():
            for file in sorted(args.output_dir.iterdir()):
                if file.is_file():
                    print(f"  - {file.name}")

        print("\n📝 Next steps for paper writing:")
        print("  1. Review the generated figures and tables")
        print("  2. Include relevant figures in your LaTeX document")
        print("  3. Reference the statistics in your Results section")
        print("  4. Use the variance data to support RQ1 claims")
        print("  5. Use the improvement data to support RQ2 claims")

    else:
        print(f"\n⚠ {total_count - success_count} analysis/analyses failed")
        print("Check the output above for error messages")
        sys.exit(1)


if __name__ == "__main__":
    main()

