#!/usr/bin/env python3
"""
ComfyUI Wan 2.2 Benchmarking Script for AMD MI300X GPUs
Tests latency with various resolutions, frame counts, and steps using Flash Attention.
"""

import json
import urllib.request
import time
import csv
import os
from datetime import datetime
from typing import Dict, Any, List
import argparse

# ComfyUI server configuration
COMFYUI_URL = "http://127.0.0.1:8188"


class ComfyUIClient:
    """Client for interacting with ComfyUI API"""

    def __init__(self, server_url: str = COMFYUI_URL):
        self.server_url = server_url

    def queue_prompt(self, prompt: Dict[str, Any]) -> str:
        """Queue a prompt and return the prompt_id"""
        data = json.dumps({"prompt": prompt}).encode('utf-8')
        req = urllib.request.Request(f"{self.server_url}/prompt", data=data)
        req.add_header('Content-Type', 'application/json')

        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read())
            return result['prompt_id']

    def get_history(self, prompt_id: str) -> Dict[str, Any]:
        """Get history for a specific prompt_id"""
        with urllib.request.urlopen(f"{self.server_url}/history/{prompt_id}") as response:
            return json.loads(response.read())

    def get_queue(self) -> Dict[str, Any]:
        """Get current queue status"""
        with urllib.request.urlopen(f"{self.server_url}/queue") as response:
            return json.loads(response.read())

    def wait_for_completion(self, prompt_id: str, poll_interval: float = 2.0) -> Dict[str, Any]:
        """Wait for a prompt to complete and return its history"""
        while True:
            history = self.get_history(prompt_id)

            # Check if prompt_id exists in history (means it's completed)
            if prompt_id in history:
                return history[prompt_id]

            # Check queue to see if it's still running
            queue_info = self.get_queue()
            queue_running = queue_info.get('queue_running', [])
            queue_pending = queue_info.get('queue_pending', [])

            # Extract prompt_ids from queue
            running_ids = [item[1] for item in queue_running]
            pending_ids = [item[1] for item in queue_pending]

            if prompt_id not in running_ids and prompt_id not in pending_ids:
                # Not in queue anymore, check history one more time
                history = self.get_history(prompt_id)
                if prompt_id in history:
                    return history[prompt_id]
                else:
                    raise RuntimeError(f"Prompt {prompt_id} disappeared from queue without completing")

            time.sleep(poll_interval)


def load_workflow_template(workflow_path: str) -> Dict[str, Any]:
    """Load the workflow template from file"""
    with open(workflow_path, 'r') as f:
        return json.load(f)


def create_benchmark_workflow(
    template: Dict[str, Any],
    width: int,
    height: int,
    length: int,
    steps: int,
    prompt: str,
    seed: int
) -> Dict[str, Any]:
    """Create a workflow with specific benchmark parameters"""
    workflow = json.loads(json.dumps(template))  # Deep copy

    # Update resolution and length (node 19 - EmptyHunyuanLatentVideo)
    workflow["19"]["inputs"]["width"] = width
    workflow["19"]["inputs"]["height"] = height
    workflow["19"]["inputs"]["length"] = length

    # Update steps for both high-noise and low-noise samplers
    # Split steps evenly between high and low noise
    high_noise_steps = steps // 2
    low_noise_start = high_noise_steps

    # Node 46 - High noise sampler
    workflow["46"]["inputs"]["steps"] = steps
    workflow["46"]["inputs"]["start_at_step"] = 0
    workflow["46"]["inputs"]["end_at_step"] = high_noise_steps
    workflow["46"]["inputs"]["noise_seed"] = seed

    # Node 14 - Low noise sampler
    workflow["14"]["inputs"]["steps"] = steps
    workflow["14"]["inputs"]["start_at_step"] = low_noise_start
    workflow["14"]["inputs"]["end_at_step"] = 10000  # Max value

    # Update prompt (node 17)
    workflow["17"]["inputs"]["text"] = prompt

    return workflow


def extract_timing_info(history: Dict[str, Any]) -> Dict[str, float]:
    """Extract timing information from history"""
    timings = {}

    if 'outputs' not in history:
        return timings

    # The history structure contains execution times
    # Look for timing in the status section
    status = history.get('status', {})

    # Check if there's a status with timing
    if 'status_str' in status:
        timings['status'] = status['status_str']

    # Get execution time from messages if available
    messages = history.get('messages', [])
    for msg in messages:
        if isinstance(msg, list) and len(msg) >= 2:
            msg_type, msg_data = msg[0], msg[1]
            if msg_type == 'execution_time':
                timings['execution_time'] = msg_data

    return timings


def run_warmup(
    client: ComfyUIClient,
    workflow_template: Dict[str, Any],
    precision: str = 'fp16',
    warmup_runs: int = 1
) -> None:
    """
    Run warm-up generation(s) to ensure models are loaded.

    This eliminates cold-start overhead from benchmark timings by:
    - Loading models into VRAM
    - Initializing Ray actors
    - Compiling kernels (Flash Attention, etc.)
    - Warming up CUDA/ROCm runtime
    """
    print("\n" + "="*80)
    print(f"WARM-UP ({warmup_runs} run{'s' if warmup_runs > 1 else ''})")
    print("="*80)
    print("Loading models and initializing GPU kernels...")
    print("(This ensures accurate benchmark timing by eliminating cold-start overhead)")

    # Use minimal configuration for fastest warm-up
    warmup_config = {
        'width': 1280,
        'height': 720,
        'length': 81,  # Shortest video length
        'steps': 20,   # Minimum steps
        'prompt': 'Test',  # Short prompt
        'seed': 0
    }

    for i in range(warmup_runs):
        if warmup_runs > 1:
            print(f"\nWarm-up run {i+1}/{warmup_runs}...")

        workflow = create_benchmark_workflow(
            workflow_template,
            width=warmup_config['width'],
            height=warmup_config['height'],
            length=warmup_config['length'],
            steps=warmup_config['steps'],
            prompt=warmup_config['prompt'],
            seed=warmup_config['seed']
        )

        start_time = time.time()
        prompt_id = client.queue_prompt(workflow)
        client.wait_for_completion(prompt_id)
        elapsed = time.time() - start_time

        if warmup_runs > 1:
            print(f"  Completed in {elapsed:.2f}s")

    print(f"\n✓ Warm-up complete! Models loaded and ready.")
    print(f"  Precision: {precision.upper()}")
    print(f"  Models are now in VRAM and kernels compiled")
    print("="*80)


def run_benchmark(
    client: ComfyUIClient,
    workflow_template: Dict[str, Any],
    config: Dict[str, Any],
    run_id: int,
    precision: str = 'fp16'
) -> Dict[str, Any]:
    """Run a single benchmark test"""
    print(f"\n{'='*80}")
    print(f"Run #{run_id}: {config['width']}x{config['height']} @ {config['length']} frames, {config['steps']} steps ({precision.upper()})")
    print(f"{'='*80}")

    # Create workflow with benchmark parameters
    workflow = create_benchmark_workflow(
        workflow_template,
        width=config['width'],
        height=config['height'],
        length=config['length'],
        steps=config['steps'],
        prompt=config['prompt'],
        seed=config.get('seed', int(time.time()))
    )

    # Queue the prompt
    print("Submitting workflow to ComfyUI...")
    start_time = time.time()
    prompt_id = client.queue_prompt(workflow)
    queue_time = time.time()
    print(f"Queued with prompt_id: {prompt_id}")

    # Wait for completion
    print("Waiting for generation to complete...")
    history = client.wait_for_completion(prompt_id)
    end_time = time.time()

    # Calculate timings
    queue_delay = queue_time - start_time
    total_time = end_time - start_time
    generation_time = total_time - queue_delay

    print(f"\n✓ Generation completed!")
    print(f"  Queue delay: {queue_delay:.2f}s")
    print(f"  Generation time: {generation_time:.2f}s")
    print(f"  Total time: {total_time:.2f}s")

    # Extract additional timing info
    timing_info = extract_timing_info(history)

    result = {
        'run_id': run_id,
        'prompt_id': prompt_id,
        'precision': precision,
        'width': config['width'],
        'height': config['height'],
        'length': config['length'],
        'steps': config['steps'],
        'prompt': config['prompt'],
        'queue_delay': queue_delay,
        'generation_time': generation_time,
        'total_time': total_time,
        'timestamp': datetime.now().isoformat(),
        **timing_info
    }

    return result


def save_results(results: List[Dict[str, Any]], output_file: str):
    """Save benchmark results to CSV"""
    if not results:
        print("No results to save")
        return

    fieldnames = [
        'run_id', 'timestamp', 'precision', 'width', 'height', 'length', 'steps',
        'queue_delay', 'generation_time', 'total_time', 'prompt_id', 'prompt'
    ]

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✓ Results saved to {output_file}")


# Benchmark configurations
# Based on recommended ranges for MI300X testing

RESOLUTION_TESTS = [
    # Standard resolutions
    {'width': 1280, 'height': 720, 'name': 'HD'},
    {'width': 1920, 'height': 1080, 'name': 'Full HD'},
    {'width': 1920, 'height': 1280, 'name': 'High Resolution'},
]

FRAME_COUNTS = [
    81,   # ~3.4s @ 24fps (short)
    161,  # ~6.7s @ 24fps (medium)
    241,  # ~10s @ 24fps (long)
]

STEP_COUNTS = [
    20,  # Fast preview
    40,  # Production quality
    50,  # High quality
]

DEFAULT_PROMPT = (
    "Epic slow-motion shot of a gaming PC with AMD Radeon graphics card rendering a fantasy world,"
    "the camera pushes through the GPU fans into the silicon where we see electrical signals racing through the chip architecture," 
    "then emerges into a vast AI data center with rows of AMD Instinct Data Center GPU accelerators processing neural networks"
)

def generate_benchmark_configs(
    resolutions: List[Dict[str, Any]],
    frame_counts: List[int],
    step_counts: List[int],
    prompt: str = DEFAULT_PROMPT,
    quick_test: bool = False
) -> List[Dict[str, Any]]:
    """Generate benchmark configurations"""
    configs = []

    if quick_test:
        # Quick test: one config for each dimension
        configs.append({
            'width': 1280,
            'height': 720,
            'length': 161,
            'steps': 40,
            'prompt': prompt,
            'seed': 12345
        })
        return configs

    # Full benchmark matrix
    for res in resolutions:
        for frames in frame_counts:
            for steps in step_counts:
                configs.append({
                    'width': res['width'],
                    'height': res['height'],
                    'length': frames,
                    'steps': steps,
                    'prompt': prompt,
                    'seed': 12345  # Fixed seed for reproducibility
                })

    return configs


def print_summary(results: List[Dict[str, Any]]):
    """Print benchmark summary statistics"""
    if not results:
        return

    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)

    # Group by configuration
    by_config = {}
    for r in results:
        key = f"{r['width']}x{r['height']}@{r['length']}frames_{r['steps']}steps"
        if key not in by_config:
            by_config[key] = []
        by_config[key].append(r['generation_time'])

    print(f"\n{'Configuration':<40} {'Gen Time (s)':<15} {'Avg (s)':<10}")
    print("-" * 80)

    for config, times in sorted(by_config.items()):
        avg_time = sum(times) / len(times)
        times_str = ", ".join([f"{t:.1f}" for t in times])
        print(f"{config:<40} {times_str:<15} {avg_time:.1f}")

    # Overall stats
    all_times = [r['generation_time'] for r in results]
    print("\n" + "-" * 80)
    print(f"Total runs: {len(results)}")
    print(f"Average generation time: {sum(all_times) / len(all_times):.2f}s")
    print(f"Min generation time: {min(all_times):.2f}s")
    print(f"Max generation time: {max(all_times):.2f}s")


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark ComfyUI Wan 2.2 on AMD MI300X GPUs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test (single config, FP16)
  python benchmark_wan22_mi300x.py --quick

  # Quick test with FP8 quantization and warm-up
  python benchmark_wan22_mi300x.py --quick --precision fp8 --warmup 1

  # Full benchmark with FP16 and warm-up (recommended)
  python benchmark_wan22_mi300x.py --warmup 2

  # Full benchmark with FP8 quantization
  python benchmark_wan22_mi300x.py --precision fp8 --warmup 1

  # Custom benchmark with explicit workflow
  python benchmark_wan22_mi300x.py --workflow workflow.json --output my_results.csv --warmup 1

  # Test specific configuration
  python benchmark_wan22_mi300x.py --width 1920 --height 1080 --length 161 --steps 50 --precision fp8 --warmup 1
        """
    )

    parser.add_argument('--workflow', type=str,
                       help='Path to workflow JSON file (API format). If not specified, auto-selects based on --precision')
    parser.add_argument('--precision', type=str, choices=['fp16', 'fp8'], default='fp16',
                       help='Model precision: fp16 (default) or fp8 (faster, lower memory)')
    parser.add_argument('--output', type=str,
                       help='Output CSV file for results (auto-generated if not specified)')
    parser.add_argument('--url', type=str, default=COMFYUI_URL,
                       help='ComfyUI server URL')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick test (single config)')
    parser.add_argument('--prompt', type=str, default=DEFAULT_PROMPT,
                       help='Text prompt for generation')
    parser.add_argument('--warmup', type=int, metavar='N', default=0,
                       help='Run N warm-up generations before benchmarking (default: 0, recommended: 1-2)')
    parser.add_argument('--no-warmup', action='store_true',
                       help='Skip automatic warm-up (not recommended for accurate timing)')

    # Custom configuration options
    parser.add_argument('--width', type=int, help='Custom width')
    parser.add_argument('--height', type=int, help='Custom height')
    parser.add_argument('--length', type=int, help='Custom frame count')
    parser.add_argument('--steps', type=int, help='Custom step count')

    args = parser.parse_args()

    # Auto-select workflow based on precision if not specified
    if args.workflow is None:
        if args.precision == 'fp8':
            args.workflow = 'example_workflows/WanT2V_MI300X_FP8_Benchmark.json'
        else:
            args.workflow = 'example_workflows/WanT2V_MI300X_Throughput_Benchmark.json'

    # Auto-generate output filename with precision if not specified
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f'benchmark_results_{args.precision}_{timestamp}.csv'

    # Load workflow template
    print(f"Model precision: {args.precision.upper()}")
    print(f"Loading workflow from: {args.workflow}")

    # For initial testing, use the provided API export
    if not os.path.exists(args.workflow):
        print(f"Warning: Workflow file not found, using inline template")
        workflow_template = {
            # Your provided workflow template would go here
            # For now, we'll load from file
        }
    else:
        workflow_template = load_workflow_template(args.workflow)

    # Create client
    client = ComfyUIClient(args.url)

    # Generate benchmark configurations
    if args.width and args.height and args.length and args.steps:
        # Custom single configuration
        configs = [{
            'width': args.width,
            'height': args.height,
            'length': args.length,
            'steps': args.steps,
            'prompt': args.prompt,
            'seed': 12345
        }]
    else:
        # Use predefined configurations
        configs = generate_benchmark_configs(
            RESOLUTION_TESTS,
            FRAME_COUNTS,
            STEP_COUNTS,
            prompt=args.prompt,
            quick_test=args.quick
        )

    print(f"\nStarting benchmark with {len(configs)} configurations")
    print(f"Results will be saved to: {args.output}")

    # Run warm-up if requested
    if args.warmup > 0 and not args.no_warmup:
        try:
            run_warmup(client, workflow_template, precision=args.precision, warmup_runs=args.warmup)
        except Exception as e:
            print(f"\n⚠ Warning: Warm-up failed: {e}")
            print("Continuing with benchmarks anyway...")
    elif not args.no_warmup and len(configs) > 1:
        # Automatic single warm-up for multi-config runs
        print("\n💡 Tip: Use --warmup 1 to ensure models are loaded before benchmarking")
        print("   (This prevents cold-start from affecting first benchmark run)")

    # Run benchmarks
    results = []
    for i, config in enumerate(configs, 1):
        try:
            result = run_benchmark(client, workflow_template, config, i, precision=args.precision)
            results.append(result)

            # Save intermediate results
            save_results(results, args.output)

        except KeyboardInterrupt:
            print("\n\nBenchmark interrupted by user")
            break
        except Exception as e:
            print(f"\n✗ Error in run #{i}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Print summary
    print_summary(results)

    # Final save
    save_results(results, args.output)

    print(f"\n{'='*80}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
