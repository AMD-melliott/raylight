#!/usr/bin/env python3
"""
Simplified Wan 2.2 Benchmark Wrapper
Directly wraps Wan's generate.py for benchmarking without modifying their code
"""

import os
import sys
import time
import csv
import subprocess
import json
from datetime import datetime
from typing import Dict, Any, List
import argparse

class Wan22Benchmark:
    def __init__(self, wan_repo_path: str, args):
        self.wan_repo = wan_repo_path
        self.args = args
        self.results = []
        self.output_file = args.output or f'benchmark_wan22_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'

    def build_wan_command(self, config: Dict[str, Any], use_torchrun: bool = False) -> List[str]:
        """Build command line for Wan generate.py"""

        # Base command
        if use_torchrun and self.args.num_gpus > 1:
            cmd = [
                'torchrun',
                '--nproc_per_node', str(self.args.num_gpus),
                os.path.join(self.wan_repo, 'generate.py')
            ]
        else:
            cmd = [
                sys.executable,
                os.path.join(self.wan_repo, 'generate.py')
            ]

        # Add required arguments
        cmd.extend([
            '--task', self.args.task,
            '--size', f"{config['width']}*{config['height']}",
            '--ckpt_dir', self.args.ckpt_dir,
            '--prompt', config['prompt']
        ])

        # Add optional image for I2V
        if self.args.image:
            cmd.extend(['--image', self.args.image])

        # Convert frames to clips (Wan uses clips, ~80 frames per clip)
        if 'length' in config:
            num_clips = max(1, config['length'] // 80)
            cmd.extend(['--num_clip', str(num_clips)])

        # Add seed for reproducibility
        if 'seed' in config:
            cmd.extend(['--seed', str(config.get('seed', 12345))])

        # Add inference steps if specified
        if 'steps' in config:
            cmd.extend(['--inference_steps', str(config['steps'])])

        # Multi-GPU settings
        if use_torchrun and self.args.num_gpus > 1:
            cmd.extend([
                '--dit_fsdp',
                '--t5_fsdp',
                '--ulysses_size', str(self.args.num_gpus)
            ])
        else:
            # Single GPU optimizations for memory
            if not self.args.no_offload:
                cmd.extend(['--offload_model', 'True'])
            if not self.args.no_fp16:
                cmd.append('--convert_model_dtype')
            if self.args.task == 'ti2v-5B' and not self.args.no_t5_cpu:
                cmd.append('--t5_cpu')

        # Add output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"./benchmark_outputs/{self.args.task}_{config['width']}x{config['height']}_{timestamp}"
        cmd.extend(['--output_dir', output_dir])

        return cmd

    def run_warmup(self, num_runs: int = 1):
        """Run warmup generations"""
        print(f"\n{'='*80}")
        print(f"WARM-UP ({num_runs} run{'s' if num_runs > 1 else ''})")
        print("="*80)

        warmup_config = {
            'width': 1280 if self.args.task != 'ti2v-5B' else 704,
            'height': 720 if self.args.task != 'ti2v-5B' else 1280,
            'length': 81,
            'steps': 20,
            'prompt': 'Warmup test generation',
            'seed': 0
        }

        for i in range(num_runs):
            print(f"\nWarm-up run {i+1}/{num_runs}...")

            cmd = self.build_wan_command(warmup_config, use_torchrun=(self.args.num_gpus > 1))

            start_time = time.time()
            try:
                if self.args.verbose:
                    print(f"Command: {' '.join(cmd)}")

                result = subprocess.run(cmd,
                                      capture_output=not self.args.verbose,
                                      text=True,
                                      cwd=self.wan_repo)

                if result.returncode != 0:
                    print(f"Warning: Warmup failed with code {result.returncode}")
                    if not self.args.verbose:
                        print(f"Error: {result.stderr}")

                elapsed = time.time() - start_time
                print(f"  Completed in {elapsed:.2f}s")

            except Exception as e:
                print(f"Warmup error: {e}")

        print("\n✓ Warm-up complete!")
        print("="*80)

    def run_benchmark(self, config: Dict[str, Any], run_id: int) -> Dict[str, Any]:
        """Run a single benchmark test"""

        print(f"\n{'='*80}")
        print(f"Run #{run_id}: {config['width']}x{config['height']} @ {config['length']} frames, {config['steps']} steps")
        if self.args.num_gpus > 1:
            print(f"Using {self.args.num_gpus} GPUs with Ulysses parallelism")
        print("="*80)

        cmd = self.build_wan_command(config, use_torchrun=(self.args.num_gpus > 1))

        if self.args.verbose:
            print(f"Command: {' '.join(cmd)}")

        print("Starting generation...")
        start_time = time.time()

        try:
            result = subprocess.run(cmd,
                                  capture_output=not self.args.verbose,
                                  text=True,
                                  cwd=self.wan_repo,
                                  timeout=self.args.timeout)

            generation_time = time.time() - start_time

            if result.returncode == 0:
                print(f"\n✓ Generation completed!")
                print(f"  Generation time: {generation_time:.2f}s")
                print(f"  Throughput: {config['length'] / generation_time:.2f} frames/s")

                return {
                    'run_id': run_id,
                    'timestamp': datetime.now().isoformat(),
                    'task': self.args.task,
                    'num_gpus': self.args.num_gpus,
                    'width': config['width'],
                    'height': config['height'],
                    'length': config['length'],
                    'steps': config['steps'],
                    'generation_time': generation_time,
                    'frames_per_second': config['length'] / generation_time,
                    'status': 'success',
                    'prompt': config['prompt'][:100]  # Truncate long prompts
                }
            else:
                print(f"✗ Generation failed with code {result.returncode}")
                if not self.args.verbose:
                    print(f"Error output: {result.stderr[:500]}")

                return {
                    'run_id': run_id,
                    'timestamp': datetime.now().isoformat(),
                    'task': self.args.task,
                    'num_gpus': self.args.num_gpus,
                    'width': config['width'],
                    'height': config['height'],
                    'length': config['length'],
                    'steps': config['steps'],
                    'generation_time': generation_time,
                    'frames_per_second': 0,
                    'status': f'failed_{result.returncode}',
                    'prompt': config['prompt'][:100]
                }

        except subprocess.TimeoutExpired:
            print(f"✗ Generation timed out after {self.args.timeout}s")
            return {
                'run_id': run_id,
                'timestamp': datetime.now().isoformat(),
                'task': self.args.task,
                'num_gpus': self.args.num_gpus,
                'width': config['width'],
                'height': config['height'],
                'length': config['length'],
                'steps': config['steps'],
                'generation_time': self.args.timeout,
                'frames_per_second': 0,
                'status': 'timeout',
                'prompt': config['prompt'][:100]
            }
        except Exception as e:
            print(f"✗ Generation failed with error: {e}")
            return None

    def save_results(self):
        """Save results to CSV"""
        if not self.results:
            return

        fieldnames = [
            'run_id', 'timestamp', 'task', 'num_gpus', 'width', 'height',
            'length', 'steps', 'generation_time', 'frames_per_second',
            'status', 'prompt'
        ]

        with open(self.output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.results)

        print(f"Results saved to {self.output_file}")

    def print_summary(self):
        """Print summary statistics"""
        if not self.results:
            return

        print("\n" + "="*80)
        print("BENCHMARK SUMMARY")
        print("="*80)

        # Filter successful runs
        successful = [r for r in self.results if r['status'] == 'success']

        if not successful:
            print("No successful runs to summarize")
            return

        # Group by configuration
        by_config = {}
        for r in successful:
            key = f"{r['width']}x{r['height']}@{r['length']}fr_{r['steps']}steps"
            if key not in by_config:
                by_config[key] = []
            by_config[key].append(r['generation_time'])

        print(f"\n{'Configuration':<40} {'Avg Time (s)':<12} {'FPS':<10} {'Runs':<6}")
        print("-" * 70)

        for config in sorted(by_config.keys()):
            times = by_config[config]
            avg_time = sum(times) / len(times)
            # Parse frames from config
            frames_str = config.split('@')[1].split('fr')[0]
            frames = int(frames_str)
            avg_fps = frames / avg_time

            print(f"{config:<40} {avg_time:<12.2f} {avg_fps:<10.2f} {len(times):<6}")

        # Overall statistics
        all_times = [r['generation_time'] for r in successful]
        print("\n" + "-" * 70)
        print(f"Total runs: {len(self.results)} ({len(successful)} successful)")
        print(f"Average generation time: {sum(all_times)/len(all_times):.2f}s")
        print(f"Min generation time: {min(all_times):.2f}s")
        print(f"Max generation time: {max(all_times):.2f}s")

        failed = len(self.results) - len(successful)
        if failed > 0:
            print(f"Failed runs: {failed}")


def generate_configs(args) -> List[Dict[str, Any]]:
    """Generate benchmark configurations"""

    default_prompt = (
        "Epic slow-motion shot of a gaming PC with AMD Radeon graphics card rendering a fantasy world, "
        "the camera pushes through the GPU fans into the silicon where we see electrical signals racing "
        "through the chip architecture, then emerges into a vast AI data center with rows of AMD Instinct "
        "Data Center GPU accelerators processing neural networks"
    )

    # Handle custom single configuration
    if args.width and args.height and args.length and args.steps:
        return [{
            'width': args.width,
            'height': args.height,
            'length': args.length,
            'steps': args.steps,
            'prompt': args.prompt or default_prompt,
            'seed': 12345
        }]

    # Quick test configuration
    if args.quick:
        if args.task == 'ti2v-5B':
            # 5B model uses different resolution
            return [{
                'width': 704,
                'height': 1280,
                'length': 81,
                'steps': 20,
                'prompt': args.prompt or default_prompt,
                'seed': 12345
            }]
        else:
            return [{
                'width': 640,
                'height': 480,
                'length': 81,
                'steps': 20,
                'prompt': args.prompt or default_prompt,
                'seed': 12345
            }]

    # Full benchmark matrix
    configs = []

    # Resolutions based on task
    if args.task == 'ti2v-5B':
        resolutions = [
            {'width': 704, 'height': 1280}  # 720P equivalent for 5B model
        ]
    else:
        resolutions = [
            {'width': 640, 'height': 480},   # 480P
            {'width': 1280, 'height': 720},  # 720P
            {'width': 1920, 'height': 1080}  # 1080P
        ]

    frame_counts = [81, 161, 241]  # ~3.4s, ~6.7s, ~10s at 24fps
    step_counts = [20, 40, 50]

    for res in resolutions:
        for frames in frame_counts:
            for steps in step_counts:
                configs.append({
                    'width': res['width'],
                    'height': res['height'],
                    'length': frames,
                    'steps': steps,
                    'prompt': args.prompt or default_prompt,
                    'seed': 12345
                })

    return configs


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark Wan 2.2 models - wrapper for native generate.py',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with T2V model
  python benchmark_wan22_native.py --wan-repo /path/to/Wan2.2 --task t2v-A14B --ckpt-dir ./Wan2.2-T2V-A14B --quick

  # Full benchmark with 8 GPUs
  python benchmark_wan22_native.py --wan-repo /path/to/Wan2.2 --task t2v-A14B --ckpt-dir ./Wan2.2-T2V-A14B --num-gpus 8 --warmup 2

  # Test 5B model on consumer GPU
  python benchmark_wan22_native.py --wan-repo /path/to/Wan2.2 --task ti2v-5B --ckpt-dir ./Wan2.2-TI2V-5B --quick

  # Custom configuration
  python benchmark_wan22_native.py --wan-repo /path/to/Wan2.2 --task t2v-A14B --ckpt-dir ./Wan2.2-T2V-A14B \\
    --width 1920 --height 1080 --length 161 --steps 50
        """
    )

    # Required arguments
    parser.add_argument('--wan-repo', type=str, required=True,
                       help='Path to Wan2.2 repository')
    parser.add_argument('--task', type=str, required=True,
                       choices=['t2v-A14B', 'i2v-A14B', 'ti2v-5B', 's2v-14B'],
                       help='Model task type')
    parser.add_argument('--ckpt-dir', type=str, required=True,
                       help='Path to model checkpoint directory')

    # Benchmark options
    parser.add_argument('--output', type=str,
                       help='Output CSV file for results')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick test (single configuration)')
    parser.add_argument('--warmup', type=int, default=0,
                       help='Number of warmup runs (recommended: 1-2)')
    parser.add_argument('--num-gpus', type=int, default=1,
                       help='Number of GPUs for parallel execution')
    parser.add_argument('--timeout', type=int, default=1800,
                       help='Timeout per generation in seconds (default: 30 min)')

    # Generation options
    parser.add_argument('--prompt', type=str,
                       help='Custom prompt for generation')
    parser.add_argument('--image', type=str,
                       help='Input image for I2V tasks')

    # Custom configuration
    parser.add_argument('--width', type=int, help='Custom width')
    parser.add_argument('--height', type=int, help='Custom height')
    parser.add_argument('--length', type=int, help='Custom frame count')
    parser.add_argument('--steps', type=int, help='Custom step count')

    # Memory optimization flags
    parser.add_argument('--no-offload', action='store_true',
                       help='Disable model offloading (single GPU mode)')
    parser.add_argument('--no-fp16', action='store_true',
                       help='Disable FP16 conversion (single GPU mode)')
    parser.add_argument('--no-t5-cpu', action='store_true',
                       help='Keep T5 on GPU for ti2v-5B task')

    # Debug options
    parser.add_argument('--verbose', action='store_true',
                       help='Show full command output')

    args = parser.parse_args()

    # Validate Wan repo exists
    if not os.path.exists(args.wan_repo):
        print(f"Error: Wan repository not found at {args.wan_repo}")
        return 1

    generate_script = os.path.join(args.wan_repo, 'generate.py')
    if not os.path.exists(generate_script):
        print(f"Error: generate.py not found in {args.wan_repo}")
        return 1

    print(f"Wan 2.2 Benchmark")
    print(f"Repository: {args.wan_repo}")
    print(f"Task: {args.task}")
    print(f"Model: {args.ckpt_dir}")
    print(f"GPUs: {args.num_gpus}")

    # Initialize benchmark runner
    benchmark = Wan22Benchmark(args.wan_repo, args)

    # Generate configurations
    configs = generate_configs(args)
    print(f"\nBenchmark configurations: {len(configs)}")

    # Run warmup if requested
    if args.warmup > 0:
        benchmark.run_warmup(args.warmup)
    elif not args.quick:
        print("\n💡 Tip: Use --warmup 1 to ensure models are loaded before benchmarking")

    # Run benchmarks
    for i, config in enumerate(configs, 1):
        try:
            result = benchmark.run_benchmark(config, i)
            if result:
                benchmark.results.append(result)
                benchmark.save_results()
        except KeyboardInterrupt:
            print("\n\nBenchmark interrupted by user")
            break
        except Exception as e:
            print(f"\nError in run {i}: {e}")
            continue

    # Print summary
    if benchmark.results:
        benchmark.print_summary()
        benchmark.save_results()

    print(f"\n{'='*80}")
    if benchmark.results and len(benchmark.results) == len(configs):
        print("BENCHMARK COMPLETE")
    else:
        print(f"BENCHMARK INCOMPLETE ({len(benchmark.results)}/{len(configs)} runs)")
    print("="*80)

    return 0


if __name__ == '__main__':
    sys.exit(main())