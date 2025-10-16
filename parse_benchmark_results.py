#!/usr/bin/env python3
"""
Parse and display benchmark results from CSV files
Displays in the same format as benchmark_wan22_mi300x.py summary
"""

import csv
import argparse
import os
import re
from typing import Dict, List, Any
from datetime import datetime


def load_csv_results(csv_file: str) -> List[Dict[str, Any]]:
    """Load benchmark results from CSV file"""
    results = []

    if not os.path.exists(csv_file):
        print(f"Error: CSV file not found: {csv_file}")
        return results

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            result = {}
            for key, value in row.items():
                if key in ['run_id', 'width', 'height', 'length', 'steps']:
                    result[key] = int(value) if value else 0
                elif key in ['queue_delay', 'generation_time', 'total_time']:
                    result[key] = float(value) if value else 0.0
                else:
                    result[key] = value
            results.append(result)

    return results


def print_summary(results: List[Dict[str, Any]], show_details: bool = False):
    """Print benchmark summary statistics matching the original format"""
    if not results:
        print("No results to display")
        return

    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)

    # Extract precision and GPU info if available
    if results[0].get('precision'):
        print(f"Precision: {results[0]['precision'].upper()}")

    # Count unique configurations
    unique_configs = set()
    for r in results:
        config = f"{r['width']}x{r['height']}@{r['length']}frames_{r['steps']}steps"
        unique_configs.add(config)

    print(f"Total runs: {len(results)}")
    print(f"Unique configurations: {len(unique_configs)}")

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

    # Calculate total GPU time (useful for throughput analysis)
    total_gpu_time = sum(all_times)
    hours = total_gpu_time / 3600
    print(f"Total GPU time: {total_gpu_time:.1f}s ({hours:.2f} hours)")

    if show_details:
        print("\n" + "="*80)
        print("DETAILED RESULTS")
        print("="*80)
        print(f"\n{'ID':<4} {'Resolution':<12} {'Frames':<8} {'Steps':<7} {'Gen Time':<10} {'Total Time':<10} {'Timestamp':<20}")
        print("-" * 80)

        for r in results:
            timestamp = r.get('timestamp', '')
            if timestamp:
                # Parse and format timestamp for readability
                try:
                    dt = datetime.fromisoformat(timestamp)
                    timestamp = dt.strftime("%Y-%m-%d %H:%M:%S")
                except:
                    pass

            print(f"{r['run_id']:<4} {r['width']}x{r['height']:<12} {r['length']:<8} {r['steps']:<7} "
                  f"{r['generation_time']:<10.1f} {r['total_time']:<10.1f} {timestamp:<20}")


def print_performance_analysis(results: List[Dict[str, Any]]):
    """Print performance analysis including scaling efficiency"""
    if not results:
        return

    print("\n" + "="*80)
    print("PERFORMANCE ANALYSIS")
    print("="*80)

    # Group by resolution and calculate metrics
    resolutions = {}
    for r in results:
        res_key = f"{r['width']}x{r['height']}"
        if res_key not in resolutions:
            resolutions[res_key] = {
                'pixels': r['width'] * r['height'],
                'configs': []
            }
        resolutions[res_key]['configs'].append(r)

    print("\nResolution Scaling:")
    print(f"{'Resolution':<15} {'Pixels':<12} {'Avg Time (s)':<15} {'Time/MPixel (s)':<15}")
    print("-" * 60)

    for res, data in sorted(resolutions.items(), key=lambda x: x[1]['pixels']):
        avg_time = sum(c['generation_time'] for c in data['configs']) / len(data['configs'])
        mpixels = data['pixels'] / 1_000_000
        time_per_mpixel = avg_time / mpixels
        print(f"{res:<15} {data['pixels']:<12,} {avg_time:<15.1f} {time_per_mpixel:<15.2f}")

    # Analyze frame count scaling
    print("\nFrame Count Scaling (at each resolution):")
    for res_key in sorted(resolutions.keys()):
        configs = resolutions[res_key]['configs']
        by_frames = {}
        for c in configs:
            if c['steps'] == 20:  # Use consistent step count for comparison
                frames = c['length']
                if frames not in by_frames:
                    by_frames[frames] = []
                by_frames[frames].append(c['generation_time'])

        if len(by_frames) > 1:
            print(f"\n  {res_key}:")
            print(f"  {'Frames':<10} {'Avg Time (s)':<15} {'Time/Frame (s)':<15}")
            print("  " + "-" * 40)
            for frames in sorted(by_frames.keys()):
                avg_time = sum(by_frames[frames]) / len(by_frames[frames])
                time_per_frame = avg_time / frames
                print(f"  {frames:<10} {avg_time:<15.1f} {time_per_frame:<15.3f}")


def main():
    parser = argparse.ArgumentParser(
        description='Parse and display benchmark results from CSV files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Display summary from CSV file
  python parse_benchmark_results.py benchmark_results_fp8_20251015_222551.csv

  # Show detailed results
  python parse_benchmark_results.py benchmark_results_fp8_20251015_222551.csv --details

  # Include performance analysis
  python parse_benchmark_results.py benchmark_results_fp8_20251015_222551.csv --analyze

  # Compare multiple CSV files
  python parse_benchmark_results.py results1.csv results2.csv --compare
        """
    )

    parser.add_argument('csv_files', nargs='+', help='CSV file(s) to parse')
    parser.add_argument('--details', '-d', action='store_true',
                       help='Show detailed results for each run')
    parser.add_argument('--analyze', '-a', action='store_true',
                       help='Show performance analysis and scaling metrics')
    parser.add_argument('--compare', '-c', action='store_true',
                       help='Compare results from multiple CSV files')

    args = parser.parse_args()

    if args.compare and len(args.csv_files) > 1:
        # Compare multiple files
        print("\n" + "="*80)
        print("BENCHMARK COMPARISON")
        print("="*80)

        all_results = {}
        for csv_file in args.csv_files:
            results = load_csv_results(csv_file)
            if results:
                # Extract filename without path and extension for label
                label = os.path.splitext(os.path.basename(csv_file))[0]
                all_results[label] = results
                print(f"\nLoaded {len(results)} results from: {csv_file}")

        if len(all_results) > 1:
            # Create comparison table
            print("\n" + "="*80)
            print("SIDE-BY-SIDE COMPARISON")
            print("="*80)

            # Find common configurations
            common_configs = None
            for label, results in all_results.items():
                configs = set()
                for r in results:
                    config = f"{r['width']}x{r['height']}@{r['length']}frames_{r['steps']}steps"
                    configs.add(config)
                if common_configs is None:
                    common_configs = configs
                else:
                    common_configs = common_configs.intersection(configs)

            if common_configs:
                print(f"\nCommon configurations found: {len(common_configs)}")
                print("\n" + "-" * 80)

                # Header
                header = f"{'Configuration':<40}"
                for label in all_results.keys():
                    # Shorten label if too long
                    short_label = label[:20] if len(label) > 20 else label
                    header += f" {short_label:<15}"
                print(header)
                print("-" * 80)

                # Sort configurations
                sorted_configs = sorted(common_configs, key=lambda x: (
                    int(re.search(r'(\d+)x(\d+)', x).group(1)) * int(re.search(r'(\d+)x(\d+)', x).group(2)) if re.search(r'(\d+)x(\d+)', x) else 0,
                    int(re.search(r'@(\d+)frames', x).group(1)) if re.search(r'@(\d+)frames', x) else 0,
                    int(re.search(r'_(\d+)steps', x).group(1)) if re.search(r'_(\d+)steps', x) else 0
                ))

                for config in sorted_configs:
                    row = f"{config:<40}"
                    for label, results in all_results.items():
                        times = []
                        for r in results:
                            r_config = f"{r['width']}x{r['height']}@{r['length']}frames_{r['steps']}steps"
                            if r_config == config:
                                times.append(r['generation_time'])
                        if times:
                            avg_time = sum(times) / len(times)
                            row += f" {avg_time:<15.1f}"
                        else:
                            row += f" {'N/A':<15}"
                    print(row)
            else:
                print("\nNo common configurations found between files")
    else:
        # Process single file or multiple files without comparison
        for csv_file in args.csv_files:
            print(f"\nProcessing: {csv_file}")
            results = load_csv_results(csv_file)

            if results:
                print_summary(results, show_details=args.details)

                if args.analyze:
                    print_performance_analysis(results)
            else:
                print(f"No valid results found in {csv_file}")

    print("\n" + "="*80)
    print("END OF REPORT")
    print("="*80)


if __name__ == '__main__':
    main()