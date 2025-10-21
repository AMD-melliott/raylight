#!/usr/bin/env python3
"""
ComfyUI Wan 2.2 Benchmarking Script for AMD MI300X GPUs
Tests latency with various resolutions, frame counts, and steps using Flash Attention.
"""

import json
import urllib.request
import urllib.error
import time
import csv
import os
import re
from datetime import datetime
from typing import Dict, Any, List, Optional
import argparse

# ComfyUI server configuration
COMFYUI_URL = "http://127.0.0.1:8188"


class ComfyUIError(Exception):
    """Base exception for ComfyUI API errors"""
    pass


class ComfyUIAPIError(ComfyUIError):
    """Exception raised when ComfyUI API returns an error response"""
    def __init__(self, status_code: int, message: str, details: Optional[Dict[str, Any]] = None):
        self.status_code = status_code
        self.message = message
        self.details = details or {}
        super().__init__(self._format_error())

    def _format_error(self) -> str:
        """Format error message for display"""
        lines = [f"ComfyUI API Error ({self.status_code}): {self.message}"]

        if self.details:
            lines.append("\nDetails:")
            for key, value in self.details.items():
                lines.append(f"  {key}: {value}")

        return "\n".join(lines)


class ComfyUIExecutionError(ComfyUIError):
    """Exception raised when workflow execution fails"""
    def __init__(self, prompt_id: str, node_id: Optional[str] = None,
                 exception_type: Optional[str] = None, exception_message: Optional[str] = None,
                 traceback: Optional[str] = None):
        self.prompt_id = prompt_id
        self.node_id = node_id
        self.exception_type = exception_type
        self.exception_message = exception_message
        self.traceback = traceback
        super().__init__(self._format_error())

    def _format_error(self) -> str:
        """Format execution error for display"""
        lines = [f"Workflow Execution Failed (prompt_id: {self.prompt_id})"]

        if self.node_id:
            lines.append(f"Failed at node: {self.node_id}")
        if self.exception_type:
            lines.append(f"Error type: {self.exception_type}")
        if self.exception_message:
            lines.append(f"Message: {self.exception_message}")
        if self.traceback:
            lines.append(f"\nTraceback:\n{self.traceback}")

        return "\n".join(lines)


class ComfyUIClient:
    """Client for interacting with ComfyUI API"""

    def __init__(self, server_url: str = COMFYUI_URL):
        self.server_url = server_url

    def queue_prompt(self, prompt: Dict[str, Any]) -> str:
        """
        Queue a prompt and return the prompt_id.

        Raises:
            ComfyUIAPIError: If the API returns an error response
            ComfyUIError: If there's a connection or communication error
        """
        data = json.dumps({"prompt": prompt}).encode('utf-8')
        req = urllib.request.Request(f"{self.server_url}/prompt", data=data)
        req.add_header('Content-Type', 'application/json')

        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                result = json.loads(response.read())

                # Check if response contains an error
                if 'error' in result:
                    error_data = result['error']
                    raise ComfyUIAPIError(
                        status_code=response.status,
                        message=error_data.get('message', 'Unknown error'),
                        details=error_data.get('details', {})
                    )

                # Check if prompt_id is present
                if 'prompt_id' not in result:
                    raise ComfyUIAPIError(
                        status_code=response.status,
                        message="API response missing 'prompt_id' field",
                        details=result
                    )

                return result['prompt_id']

        except urllib.error.HTTPError as e:
            # HTTP error (4xx, 5xx)
            try:
                error_body = json.loads(e.read().decode('utf-8'))
                message = error_body.get('error', {}).get('message', str(e))
                details = error_body.get('error', {}).get('details', {})
            except:
                message = str(e)
                details = {}

            raise ComfyUIAPIError(
                status_code=e.code,
                message=message,
                details=details
            ) from e

        except urllib.error.URLError as e:
            raise ComfyUIError(
                f"Cannot connect to ComfyUI at {self.server_url}. "
                f"Ensure ComfyUI is running. Error: {e.reason}"
            ) from e

        except TimeoutError as e:
            raise ComfyUIError(
                f"Timeout connecting to ComfyUI at {self.server_url}"
            ) from e

        except json.JSONDecodeError as e:
            raise ComfyUIError(
                f"Invalid JSON response from ComfyUI API: {e}"
            ) from e

    def get_history(self, prompt_id: str) -> Dict[str, Any]:
        """Get history for a specific prompt_id"""
        try:
            with urllib.request.urlopen(f"{self.server_url}/history/{prompt_id}", timeout=10) as response:
                return json.loads(response.read())
        except urllib.error.URLError as e:
            raise ComfyUIError(f"Failed to get history: {e}") from e

    def get_queue(self) -> Dict[str, Any]:
        """Get current queue status"""
        try:
            with urllib.request.urlopen(f"{self.server_url}/queue", timeout=10) as response:
                return json.loads(response.read())
        except urllib.error.URLError as e:
            raise ComfyUIError(f"Failed to get queue status: {e}") from e

    def wait_for_completion(self, prompt_id: str, poll_interval: float = 2.0) -> Dict[str, Any]:
        """
        Wait for a prompt to complete and return its history.

        Raises:
            ComfyUIExecutionError: If the workflow execution fails
            ComfyUIError: If there's a communication error
        """
        while True:
            history = self.get_history(prompt_id)

            # Check if prompt_id exists in history (means it's completed)
            if prompt_id in history:
                result = history[prompt_id]

                # Check if execution failed
                status = result.get('status', {})
                if status.get('status_str') == 'error' or 'error' in status:
                    # Extract error information
                    error_info = status.get('error', {})
                    messages = result.get('messages', [])

                    # Try to find detailed error in messages
                    exception_type = None
                    exception_message = None
                    traceback = None
                    node_id = error_info.get('node_id')

                    for msg in messages:
                        if isinstance(msg, list) and len(msg) >= 2:
                            msg_type, msg_data = msg[0], msg[1]
                            if msg_type == 'execution_error':
                                exception_type = msg_data.get('exception_type')
                                exception_message = msg_data.get('exception_message')
                                traceback = msg_data.get('traceback')
                                node_id = node_id or msg_data.get('node_id')

                    # Fallback to error dict if messages don't have details
                    if not exception_message and 'exception_message' in error_info:
                        exception_message = error_info['exception_message']
                    if not exception_type and 'exception_type' in error_info:
                        exception_type = error_info['exception_type']

                    raise ComfyUIExecutionError(
                        prompt_id=prompt_id,
                        node_id=node_id,
                        exception_type=exception_type,
                        exception_message=exception_message or "Unknown execution error",
                        traceback=traceback
                    )

                return result

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
                    result = history[prompt_id]
                    # Check for errors even on this final check
                    status = result.get('status', {})
                    if status.get('status_str') == 'error':
                        error_info = status.get('error', {})
                        raise ComfyUIExecutionError(
                            prompt_id=prompt_id,
                            node_id=error_info.get('node_id'),
                            exception_message=error_info.get('exception_message', 'Unknown error')
                        )
                    return result
                else:
                    raise ComfyUIError(f"Prompt {prompt_id} disappeared from queue without completing")

            time.sleep(poll_interval)


def load_workflow_template(workflow_path: str) -> Dict[str, Any]:
    """
    Load the workflow template from file.

    Raises:
        FileNotFoundError: If workflow file doesn't exist
        ComfyUIError: If workflow file is invalid JSON
    """
    try:
        with open(workflow_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Workflow file not found: {workflow_path}\n"
            f"Expected location: {os.path.abspath(workflow_path)}"
        )
    except json.JSONDecodeError as e:
        raise ComfyUIError(f"Invalid JSON in workflow file {workflow_path}: {e}")


def create_benchmark_workflow(
    template: Dict[str, Any],
    width: int,
    height: int,
    length: int,
    steps: int,
    prompt: str,
    seed: int,
    num_gpus: int = 8
) -> Dict[str, Any]:
    """Create a workflow with specific benchmark parameters"""
    workflow = json.loads(json.dumps(template))  # Deep copy

    # Update GPU configuration (node 38 - RayInitializer)
    if "38" in workflow:
        workflow["38"]["inputs"]["GPU"] = num_gpus
        # Set ulysses_degree to match GPU count for optimal parallelism
        workflow["38"]["inputs"]["ulysses_degree"] = num_gpus

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
    warmup_runs: int = 1,
    num_gpus: int = 8,
    concise: bool = False
) -> None:
    """
    Run warm-up generation(s) to ensure models are loaded.

    This eliminates cold-start overhead from benchmark timings by:
    - Loading models into VRAM
    - Initializing Ray actors
    - Compiling kernels (Flash Attention, etc.)
    - Warming up CUDA/ROCm runtime

    Raises:
        ComfyUIError: If warm-up generation fails
    """
    if concise:
        print(f"\nWarm-up: {warmup_runs} run{'s' if warmup_runs > 1 else ''}... ", end="", flush=True)
    else:
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
        if not concise and warmup_runs > 1:
            print(f"\nWarm-up run {i+1}/{warmup_runs}...")

        workflow = create_benchmark_workflow(
            workflow_template,
            width=warmup_config['width'],
            height=warmup_config['height'],
            length=warmup_config['length'],
            steps=warmup_config['steps'],
            prompt=warmup_config['prompt'],
            seed=warmup_config['seed'],
            num_gpus=num_gpus
        )

        start_time = time.time()
        prompt_id = client.queue_prompt(workflow)
        client.wait_for_completion(prompt_id)
        elapsed = time.time() - start_time

        if not concise and warmup_runs > 1:
            print(f"  Completed in {elapsed:.2f}s")

    if concise:
        print(f"✓ {elapsed:.2f}s")
    else:
        print(f"\n✓ Warm-up complete! Models loaded and ready.")
        print(f"  Precision: {precision.upper()}")
        print(f"  GPU count: {num_gpus}")
        print(f"  Models are now in VRAM and kernels compiled")
        print("="*80)


def run_benchmark(
    client: ComfyUIClient,
    workflow_template: Dict[str, Any],
    config: Dict[str, Any],
    run_id: int,
    precision: str = 'fp16',
    num_gpus: int = 8,
    concise: bool = False
) -> Dict[str, Any]:
    """Run a single benchmark test"""
    if concise:
        # Concise output for singleton runs
        print(f"\n[{precision.upper()}, {num_gpus} GPUs] {config['width']}x{config['height']} @ {config['length']} frames, {config['steps']} steps")
    else:
        print(f"\n{'='*80}")
        print(f"Run #{run_id}: {config['width']}x{config['height']} @ {config['length']} frames, {config['steps']} steps ({precision.upper()}, {num_gpus} GPUs)")
        print(f"{'='*80}")

    # Create workflow with benchmark parameters
    workflow = create_benchmark_workflow(
        workflow_template,
        width=config['width'],
        height=config['height'],
        length=config['length'],
        steps=config['steps'],
        prompt=config['prompt'],
        seed=config.get('seed', int(time.time())),
        num_gpus=num_gpus
    )

    # Queue the prompt
    if concise:
        print("Submitting to ComfyUI... ", end="", flush=True)
    else:
        print("Submitting workflow to ComfyUI...")
    start_time = time.time()
    prompt_id = client.queue_prompt(workflow)
    queue_time = time.time()
    if concise:
        print(f"queued ({prompt_id[:8]}...)")
        print("Generating... ", end="", flush=True)
    else:
        print(f"Queued with prompt_id: {prompt_id}")
        print("Waiting for generation to complete...")

    # Wait for completion
    history = client.wait_for_completion(prompt_id)
    end_time = time.time()

    # Calculate timings
    queue_delay = queue_time - start_time
    total_time = end_time - start_time
    generation_time = total_time - queue_delay

    if concise:
        print(f"✓ {generation_time:.2f}s")
    else:
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


def save_results(results: List[Dict[str, Any]], output_file: str, concise: bool = False):
    """Save benchmark results to CSV"""
    if not results:
        if not concise:
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

    if concise:
        print(f"Saved: {output_file}")
    else:
        print(f"\n✓ Results saved to {output_file}")


# Benchmark configurations
# Based on recommended ranges for MI300X testing

RESOLUTION_TESTS = [
    # Standard resolutions
    {'width': 640, 'height': 480, 'name': '480P'},
    {'width': 1280, 'height': 720, 'name': '720P'},
    {'width': 1920, 'height': 1080, 'name': '1080P'},
    #{'width': 1920, 'height': 1280, 'name': 'High Resolution'},
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
    #"3D metallic letters 'AMD' floating above 'TME' in bold typography, with electric blue lightning arcing between the letters, dramatic rim lighting, sparks and embers rising upward, dark background with volumetric fog, cinematic lighting, photorealistic rendering"
    "Glass letters 'AMD' with internal RGB lighting suspended above mirror-polished 'TME', complex light refraction and caustics patterns on surrounding surfaces, studio lighting setup, slow camera orbit, crystal-clear reflections"
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
            'width': 640,
            'height': 480,
            'length': 81,
            'steps': 20,
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


def print_summary(results: List[Dict[str, Any]], concise: bool = False):
    """Print benchmark summary statistics"""
    if not results:
        return

    # Skip summary for singleton runs in concise mode
    if concise and len(results) == 1:
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

    # Sort by resolution (total pixels), then frames, then steps
    def sort_key(item):
        config_str = item[0]
        # Parse config string: "1280x720@161frames_40steps"
        match = re.match(r'(\d+)x(\d+)@(\d+)frames_(\d+)steps', config_str)
        if match:
            width = int(match.group(1))
            height = int(match.group(2))
            frames = int(match.group(3))
            steps = int(match.group(4))
            # Sort by total pixels (resolution), then frames, then steps
            return (width * height, frames, steps)
        return (0, 0, 0)  # Fallback for unparseable strings

    for config, times in sorted(by_config.items(), key=sort_key):
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

  # Quick test with concise output (ideal for singleton runs)
  python benchmark_wan22_mi300x.py --quick --concise --warmup 1

  # Quick test with 4 GPUs and warm-up
  python benchmark_wan22_mi300x.py --quick --num-gpus 4 --warmup 1

  # Quick test with FP8 quantization and warm-up (concise)
  python benchmark_wan22_mi300x.py --quick --precision fp8 --warmup 1 --concise

  # Full benchmark with FP16 and warm-up (recommended)
  python benchmark_wan22_mi300x.py --warmup 2

  # Full benchmark with 4 GPUs
  python benchmark_wan22_mi300x.py --num-gpus 4 --warmup 1

  # Full benchmark with FP8 quantization
  python benchmark_wan22_mi300x.py --precision fp8 --warmup 1

  # Custom benchmark with explicit workflow
  python benchmark_wan22_mi300x.py --workflow workflow.json --output my_results.csv --warmup 1

  # Test specific configuration with 2 GPUs
  python benchmark_wan22_mi300x.py --width 1920 --height 1080 --length 161 --steps 50 --num-gpus 2 --precision fp8 --warmup 1
        """
    )

    parser.add_argument('--workflow', type=str,
                       help='Path to workflow JSON file (API format). If not specified, auto-selects based on --precision')
    parser.add_argument('--precision', type=str, choices=['fp16', 'fp8', 'fp8_scaled'], default='fp16',
                       help='Model precision: fp16 (default) or fp8 (faster, lower memory)')
    parser.add_argument('--output', type=str,
                       help='Output CSV file for results (auto-generated if not specified)')
    parser.add_argument('--url', type=str, default=COMFYUI_URL,
                       help='ComfyUI server URL')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick test (single config)')
    parser.add_argument('--concise', action='store_true',
                       help='Use concise output format (ideal for singleton benchmarks)')
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
    parser.add_argument('--num-gpus', type=int, default=8,
                       help='Number of GPUs to use for parallelization (default: 8)')

    args = parser.parse_args()

    # Auto-select workflow based on precision if not specified
    if args.workflow is None:
        if args.precision == 'fp8':
            args.workflow = 'example_workflows/WanT2V_MI300X_FP8_Benchmark.json'
        elif args.precision == 'fp8_scaled':
            args.workflow = 'example_workflows/WanT2V_MI300X_FP8_Benchmark_scaled.json'
        else:
            args.workflow = 'example_workflows/WanT2V_MI300X_Throughput_Benchmark.json'

    # Auto-generate output filename with precision if not specified
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f'benchmark_results_{args.precision}_{timestamp}.csv'

    # Load workflow template
    if not args.concise:
        print(f"Model precision: {args.precision.upper()}")
        print(f"GPU configuration: {args.num_gpus} GPUs")
        print(f"Loading workflow from: {args.workflow}")
    else:
        print(f"Config: {args.precision.upper()}, {args.num_gpus} GPUs, {args.workflow}")

    try:
        workflow_template = load_workflow_template(args.workflow)
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("\nAvailable workflow templates:")
        print(f"  FP16: example_workflows/WanT2V_MI300X_Throughput_Benchmark.json")
        print(f"  FP8:  example_workflows/WanT2V_MI300X_FP8_Benchmark.json")
        print(f"  FP8:  example_workflows/WanT2V_MI300X_FP8_Benchmark_scaled.json")
        return 1
    except ComfyUIError as e:
        print(f"\n✗ Error loading workflow: {e}")
        return 1

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

    if not args.concise:
        print(f"\nStarting benchmark with {len(configs)} configurations")
        print(f"Results will be saved to: {args.output}")

    # Run warm-up if requested
    if args.warmup > 0 and not args.no_warmup:
        try:
            run_warmup(client, workflow_template, precision=args.precision,
                      warmup_runs=args.warmup, num_gpus=args.num_gpus, concise=args.concise)
        except ComfyUIAPIError as e:
            print(f"\n✗ Warm-up failed with API error:")
            print(f"{e}")
            print("\nPossible causes:")
            print("  - Models not downloaded to ComfyUI/models/diffusion_models/")
            print("  - Incorrect workflow configuration")
            print("  - ComfyUI custom nodes not installed")
            print("\nFix the error and try again.")
            return 1
        except ComfyUIExecutionError as e:
            print(f"\n✗ Warm-up failed with execution error:")
            print(f"{e}")
            print("\nPossible causes:")
            print("  - GPU out of memory")
            print("  - Ray actor initialization failed")
            print("  - Model loading error")
            print("\nFix the error and try again.")
            return 1
        except ComfyUIError as e:
            print(f"\n✗ Warm-up failed:")
            print(f"{e}")
            print("\nEnsure ComfyUI is running at {args.url}")
            return 1
    elif not args.no_warmup and len(configs) > 1 and not args.concise:
        # Automatic single warm-up for multi-config runs
        print("\n💡 Tip: Use --warmup 1 to ensure models are loaded before benchmarking")
        print("   (This prevents cold-start from affecting first benchmark run)")

    # Run benchmarks
    results = []
    for i, config in enumerate(configs, 1):
        try:
            result = run_benchmark(client, workflow_template, config, i,
                                  precision=args.precision, num_gpus=args.num_gpus,
                                  concise=args.concise)
            results.append(result)

            # Save intermediate results
            save_results(results, args.output, concise=args.concise)

        except KeyboardInterrupt:
            print("\n\nBenchmark interrupted by user")
            break
        except ComfyUIAPIError as e:
            print(f"\n✗ Run #{i} failed with API error:")
            print(f"{e}")
            print("\nBenchmark aborted. Fix the error and try again.")
            break
        except ComfyUIExecutionError as e:
            print(f"\n✗ Run #{i} failed with execution error:")
            print(f"{e}")
            print("\nBenchmark aborted. Fix the error and try again.")
            break
        except ComfyUIError as e:
            print(f"\n✗ Run #{i} failed:")
            print(f"{e}")
            print("\nBenchmark aborted. Fix the error and try again.")
            break
        except Exception as e:
            print(f"\n✗ Run #{i} failed with unexpected error:")
            print(f"{e}")
            import traceback
            traceback.print_exc()
            print("\nBenchmark aborted. Fix the error and try again.")
            break

    # Print summary if we have any results
    if results:
        print_summary(results, concise=args.concise)

        # Final save
        save_results(results, args.output, concise=args.concise)

    if not args.concise:
        print(f"\n{'='*80}")
        if len(results) == len(configs):
            print("BENCHMARK COMPLETE")
        else:
            print(f"BENCHMARK INCOMPLETE ({len(results)}/{len(configs)} runs completed)")
        print(f"{'='*80}")

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
