#!/usr/bin/env python3
"""
Benchmark different attention backends for Wan 2.2 on MI300X
Automatically tests all available backends and compares performance.
"""

import json
import subprocess
import sys
import os
from datetime import datetime
from pathlib import Path

# Attention backends to test (from src/raylight/distributed_modules/attention.py)
BACKENDS = [
    "FLASH_ATTN",
    "SAGE_FP8",
    "SAGE_FP16_CUDA",
    "SAGE_FP16_TRITON",
    "TORCH",
]

# Test configuration
DEFAULT_CONFIG = {
    'width': 1280,
    'height': 720,
    'length': 161,
    'steps': 40,
}

WORKFLOW_TEMPLATE = "example_workflows/WanT2V_MI300X_Throughput_Benchmark.json"


def modify_workflow_backend(workflow_path: str, backend: str, output_path: str):
    """Modify workflow to use specific attention backend"""
    with open(workflow_path, 'r') as f:
        workflow = json.load(f)

    # Update RayInitializer node (node 38)
    if "38" in workflow:
        workflow["38"]["inputs"]["XFuser_attention"] = backend
    else:
        print(f"Warning: Could not find RayInitializer node in workflow")
        return False

    # Save modified workflow
    with open(output_path, 'w') as f:
        json.dump(workflow, f, indent=2)

    return True


def run_benchmark_for_backend(backend: str, config: dict) -> dict:
    """Run benchmark for a specific backend"""
    print(f"\n{'='*80}")
    print(f"Testing Backend: {backend}")
    print(f"{'='*80}\n")

    # Create temporary workflow with this backend
    temp_workflow = f"/tmp/workflow_{backend}.json"
    if not modify_workflow_backend(WORKFLOW_TEMPLATE, backend, temp_workflow):
        return None

    # Build benchmark command
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv = f"results_{backend}_{timestamp}.csv"

    cmd = [
        sys.executable,
        "benchmark_wan22_mi300x.py",
        "--workflow", temp_workflow,
        "--output", output_csv,
        "--width", str(config['width']),
        "--height", str(config['height']),
        "--length", str(config['length']),
        "--steps", str(config['steps']),
    ]

    print(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )

        if result.returncode != 0:
            print(f"Error running benchmark for {backend}:")
            print(result.stderr)
            return None

        # Parse the generation time from output
        # Look for "Generation time: XX.XXs"
        gen_time = None
        for line in result.stdout.split('\n'):
            if 'Generation time:' in line:
                try:
                    gen_time = float(line.split(':')[1].strip().rstrip('s'))
                except:
                    pass

        return {
            'backend': backend,
            'generation_time': gen_time,
            'output_file': output_csv,
            'stdout': result.stdout,
        }

    except subprocess.TimeoutExpired:
        print(f"Timeout running benchmark for {backend}")
        return None
    except Exception as e:
        print(f"Exception running benchmark for {backend}: {e}")
        return None
    finally:
        # Cleanup temp workflow
        if os.path.exists(temp_workflow):
            os.remove(temp_workflow)


def print_comparison_table(results: list):
    """Print comparison table of all backends"""
    print("\n" + "="*80)
    print("ATTENTION BACKEND COMPARISON")
    print("="*80)

    # Filter successful results
    successful = [r for r in results if r and r['generation_time']]

    if not successful:
        print("No successful benchmark runs")
        return

    # Sort by generation time
    successful.sort(key=lambda x: x['generation_time'])

    # Print table
    print(f"\n{'Rank':<6} {'Backend':<25} {'Time (s)':<12} {'Relative Speed':<15}")
    print("-" * 80)

    baseline = successful[0]['generation_time']

    for i, result in enumerate(successful, 1):
        gen_time = result['generation_time']
        speedup = baseline / gen_time
        relative = f"{speedup:.2f}x"

        print(f"{i:<6} {result['backend']:<25} {gen_time:<12.2f} {relative:<15}")

    # Summary
    print("\n" + "-" * 80)
    print(f"Fastest: {successful[0]['backend']} ({successful[0]['generation_time']:.2f}s)")
    if len(successful) > 1:
        slowest = successful[-1]
        improvement = (slowest['generation_time'] - baseline) / baseline * 100
        print(f"Slowest: {slowest['backend']} ({slowest['generation_time']:.2f}s)")
        print(f"Performance delta: {improvement:.1f}% slower than fastest")

    # Save comparison
    comparison_file = f"backend_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(comparison_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("ATTENTION BACKEND COMPARISON\n")
        f.write("="*80 + "\n\n")

        f.write(f"Test Configuration:\n")
        for key, value in DEFAULT_CONFIG.items():
            f.write(f"  {key}: {value}\n")
        f.write(f"\n{'Rank':<6} {'Backend':<25} {'Time (s)':<12} {'Relative Speed':<15}\n")
        f.write("-" * 80 + "\n")

        for i, result in enumerate(successful, 1):
            gen_time = result['generation_time']
            speedup = baseline / gen_time
            relative = f"{speedup:.2f}x"
            f.write(f"{i:<6} {result['backend']:<25} {gen_time:<12.2f} {relative:<15}\n")

        f.write("\n" + "-" * 80 + "\n")
        f.write(f"Fastest: {successful[0]['backend']} ({successful[0]['generation_time']:.2f}s)\n")
        if len(successful) > 1:
            slowest = successful[-1]
            improvement = (slowest['generation_time'] - baseline) / baseline * 100
            f.write(f"Slowest: {slowest['backend']} ({slowest['generation_time']:.2f}s)\n")
            f.write(f"Performance delta: {improvement:.1f}% slower than fastest\n")

    print(f"\nComparison saved to: {comparison_file}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Benchmark all attention backends for Wan 2.2',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script will test all available attention backends and compare their performance.

Backends tested:
  - FLASH_ATTN: FlashAttention-2 (default)
  - SAGE_FP8: SAGE with FP8 quantization (likely fastest on MI300X)
  - SAGE_FP16_CUDA: SAGE FP16 CUDA implementation
  - SAGE_FP16_TRITON: SAGE FP16 Triton implementation
  - TORCH: PyTorch native attention (baseline)

Example:
  python benchmark_attention_backends.py
  python benchmark_attention_backends.py --backends FLASH_ATTN SAGE_FP8 TORCH
  python benchmark_attention_backends.py --steps 50 --length 241
        """
    )

    parser.add_argument('--backends', nargs='+', choices=BACKENDS,
                       default=BACKENDS,
                       help='Backends to test (default: all)')
    parser.add_argument('--width', type=int, default=DEFAULT_CONFIG['width'],
                       help=f"Video width (default: {DEFAULT_CONFIG['width']})")
    parser.add_argument('--height', type=int, default=DEFAULT_CONFIG['height'],
                       help=f"Video height (default: {DEFAULT_CONFIG['height']})")
    parser.add_argument('--length', type=int, default=DEFAULT_CONFIG['length'],
                       help=f"Frame count (default: {DEFAULT_CONFIG['length']})")
    parser.add_argument('--steps', type=int, default=DEFAULT_CONFIG['steps'],
                       help=f"Denoising steps (default: {DEFAULT_CONFIG['steps']})")

    args = parser.parse_args()

    # Update config from args
    config = {
        'width': args.width,
        'height': args.height,
        'length': args.length,
        'steps': args.steps,
    }

    # Check prerequisites
    if not os.path.exists(WORKFLOW_TEMPLATE):
        print(f"Error: Workflow template not found: {WORKFLOW_TEMPLATE}")
        print("Please ensure you're in the raylight directory")
        sys.exit(1)

    if not os.path.exists("benchmark_wan22_mi300x.py"):
        print("Error: benchmark_wan22_mi300x.py not found")
        print("Please ensure you're in the raylight directory")
        sys.exit(1)

    print("="*80)
    print("ATTENTION BACKEND BENCHMARK")
    print("="*80)
    print(f"\nTesting {len(args.backends)} backends with configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print(f"\nBackends to test: {', '.join(args.backends)}")
    print("\nThis may take a while (several minutes per backend)...")

    # Run benchmarks
    results = []
    for i, backend in enumerate(args.backends, 1):
        print(f"\n[{i}/{len(args.backends)}] Testing {backend}...")
        result = run_benchmark_for_backend(backend, config)
        if result:
            results.append(result)
        else:
            print(f"Skipping {backend} due to errors")

    # Print comparison
    print_comparison_table(results)

    print("\n" + "="*80)
    print("BENCHMARK COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
