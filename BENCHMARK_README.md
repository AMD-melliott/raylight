# Wan 2.2 Benchmarking Guide for AMD MI300X

This guide covers benchmarking Wan 2.2 T2V model on AMD MI300X GPUs using the raylight parallelism framework.

## Overview

The `benchmark_wan22_mi300x.py` script automates testing of video generation performance across different:
- **Resolutions**: 720p (HD), 1080p (Full HD), 1920x1280 (High-res)
- **Frame counts**: 81, 161, 241 frames
- **Step counts**: 20, 40, 50 steps

## Prerequisites

1. **ComfyUI running with raylight installed**
   ```bash
   # ComfyUI should be accessible at http://127.0.0.1:8188
   ```

2. **Wan 2.2 T2V models downloaded**
   - `Wan2.2/wan2.2_t2v_high_noise_14B_fp16.safetensors`
   - `Wan2.2/wan2.2_t2v_low_noise_14B_fp16.safetensors`
   - `umt5_xxl_fp16.safetensors` (CLIP)
   - `wan_2.1_vae.safetensors` (VAE)

3. **8x AMD MI300X GPUs** (or adjust GPU count in workflow)

## Understanding Wan 2.2 Model Architecture

### MoE (Mixture of Experts) Design

Wan 2.2 uses **two expert models** that automatically switch during generation:

- **High-Noise Expert** (14B params): Handles early denoising steps (0-20)
  - Focuses on overall layout and composition
  - Active when noise level is high

- **Low-Noise Expert** (14B params): Handles late denoising steps (20-40)
  - Refines details and quality
  - Active when noise level is low

**Total model size**: ~27B parameters
**Active per step**: ~14B parameters (only one expert active at a time)

### Why You Need BOTH Models

You **cannot use just one model** - the workflow requires both:
1. High-noise model processes first half of steps
2. Output is passed to low-noise model
3. Low-noise model refines for second half of steps

This is NOT optional - it's how Wan 2.2 achieves better quality than Wan 2.1 at the same compute cost.

## Workflow Configuration

The provided workflow (`WanT2V_MI300X_Throughput_Benchmark.json`) is configured for:

- **8 GPUs** with **USP (Sequence Parallel)** mode
- **Flash Attention** backend
- **NO FSDP** (unnecessary with 192GB VRAM per GPU)

Key settings in workflow:
```json
"38": {  // RayInitializer node
  "GPU": 8,
  "ulysses_degree": 8,     // Split sequence across 8 GPUs
  "ring_degree": 1,
  "FSDP": false,           // Skip weight sharding
  "XFuser_attention": "FLASH_ATTN"
}
```

### Model Loading Nodes

```json
"45": {  // High-noise model
  "unet_name": "Wan2.2/wan2.2_t2v_high_noise_14B_fp16.safetensors"
}

"8": {   // Low-noise model
  "unet_name": "Wan2.2/wan2.2_t2v_low_noise_14B_fp16.safetensors"
}
```

### Sampler Split (MoE switching)

```json
"46": {  // High-noise sampler
  "start_at_step": 0,
  "end_at_step": 20,       // First half of steps
  "return_with_leftover_noise": "enable"
}

"14": {  // Low-noise sampler
  "start_at_step": 20,     // Second half of steps
  "end_at_step": 10000
}
```

## Running Benchmarks

### Quick Test (Single Configuration)

```bash
python benchmark_wan22_mi300x.py --quick
```

This runs a single test at:
- 1280x720 resolution
- 161 frames (~6.7s video)
- 40 steps

### Full Benchmark Suite

```bash
python benchmark_wan22_mi300x.py --workflow example_workflows/WanT2V_MI300X_Throughput_Benchmark.json
```

This runs **27 configurations** (3 resolutions × 3 frame counts × 3 step counts):

| Resolution | Frames | Steps | Total Configs |
|------------|--------|-------|---------------|
| 1280x720   | 81, 161, 241 | 20, 40, 50 | 9 |
| 1920x1080  | 81, 161, 241 | 20, 40, 50 | 9 |
| 1920x1280  | 81, 161, 241 | 20, 40, 50 | 9 |

### Custom Configuration

Test a specific configuration:

```bash
python benchmark_wan22_mi300x.py \
  --width 1920 \
  --height 1080 \
  --length 161 \
  --steps 50
```

### Custom Prompt

```bash
python benchmark_wan22_mi300x.py \
  --quick \
  --prompt "A futuristic city at sunset, flying cars, neon lights"
```

## Understanding the Output

### Console Output

```
================================================================================
Run #1: 1280x720 @ 161 frames, 40 steps
================================================================================
Submitting workflow to ComfyUI...
Queued with prompt_id: abc123...
Waiting for generation to complete...

✓ Generation completed!
  Queue delay: 0.15s
  Generation time: 45.32s
  Total time: 45.47s
```

### CSV Results File

The script generates `benchmark_results_YYYYMMDD_HHMMSS.csv` with columns:

| Column | Description |
|--------|-------------|
| `run_id` | Sequential test number |
| `timestamp` | ISO 8601 timestamp |
| `width` | Video width in pixels |
| `height` | Video height in pixels |
| `length` | Frame count |
| `steps` | Denoising steps |
| `queue_delay` | Time to queue (seconds) |
| `generation_time` | Pure generation time (seconds) |
| `total_time` | Queue + generation (seconds) |
| `prompt_id` | ComfyUI prompt identifier |
| `prompt` | Text prompt used |

### Summary Statistics

At the end of the run:

```
================================================================================
BENCHMARK SUMMARY
================================================================================

Configuration                             Gen Time (s)    Avg (s)
--------------------------------------------------------------------------------
1280x720@81frames_20steps                 23.5           23.5
1280x720@161frames_40steps                45.3           45.3
1920x1080@161frames_50steps               72.1           72.1
...

--------------------------------------------------------------------------------
Total runs: 27
Average generation time: 52.34s
Min generation time: 23.45s
Max generation time: 95.67s
```

## Recommended Benchmark Strategy

### Phase 1: Latency Testing (Single Generation)

Focus: How fast can you generate ONE video?

```bash
# Test different resolutions at fixed frames/steps
python benchmark_wan22_mi300x.py \
  --workflow example_workflows/WanT2V_MI300X_Throughput_Benchmark.json \
  --width 1280 --height 720 --length 161 --steps 50

python benchmark_wan22_mi300x.py \
  --width 1920 --height 1080 --length 161 --steps 50

python benchmark_wan22_mi300x.py \
  --width 1920 --height 1280 --length 161 --steps 50
```

**Metric**: Seconds per generation

### Phase 2: Scaling Testing

Test how generation time scales with:

**A. Frame count (fixed resolution/steps)**
```bash
for frames in 81 161 241; do
  python benchmark_wan22_mi300x.py \
    --width 1280 --height 720 --length $frames --steps 40
done
```

**B. Step count (fixed resolution/frames)**
```bash
for steps in 20 30 40 50; do
  python benchmark_wan22_mi300x.py \
    --width 1280 --height 720 --length 161 --steps $steps
done
```

**C. Resolution (fixed frames/steps)**
```bash
# Already covered in Phase 1
```

### Phase 3: Attention Backend Comparison

Test different attention implementations:

**Note**: You'll need to modify the workflow JSON for each backend.

Edit `example_workflows/WanT2V_MI300X_Throughput_Benchmark.json`:

```json
"38": {
  "inputs": {
    "XFuser_attention": "FLASH_ATTN"  // Change this line
  }
}
```

Available backends (from `src/raylight/distributed_modules/attention.py`):
- `FLASH_ATTN` - FlashAttention-2 (default)
- `SAGE_FP8` - SAGE with FP8 (likely fastest on MI300X)
- `SAGE_FP16_CUDA` - SAGE FP16 CUDA
- `SAGE_FP16_TRITON` - SAGE FP16 Triton
- `TORCH` - PyTorch native (baseline)

**Expected performance on MI300X** (based on CDNA architecture):
1. `SAGE_FP8` (fastest - uses Tensor Cores)
2. `SAGE_FP16_CUDA`
3. `FLASH_ATTN`
4. `SAGE_FP16_TRITON`
5. `TORCH` (slowest)

## Analyzing Results

### Import to pandas

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('benchmark_results_20251014_123456.csv')

# Group by configuration
grouped = df.groupby(['width', 'height', 'length', 'steps'])['generation_time'].mean()
print(grouped)

# Plot scaling with frame count
df_720p = df[(df['width'] == 1280) & (df['height'] == 720) & (df['steps'] == 40)]
df_720p.plot(x='length', y='generation_time', kind='scatter')
plt.xlabel('Frame Count')
plt.ylabel('Generation Time (s)')
plt.title('Wan 2.2 Scaling: Frame Count vs Generation Time (720p, 40 steps)')
plt.show()
```

### Expected Scaling Characteristics

**USP Mode (Sequence Parallel)**:
- **Near-linear speedup** with GPU count for latency
- 8 GPUs should be ~7-7.5× faster than 1 GPU (not quite 8× due to communication overhead)

**Frame Count Scaling**:
- Should scale roughly linearly (2× frames ≈ 2× time)

**Step Count Scaling**:
- Should scale linearly (2× steps ≈ 2× time)

**Resolution Scaling**:
- Scales with pixel count (1920x1080 is 2.25× pixels of 1280x720)
- May not be perfectly linear due to memory access patterns

## Troubleshooting

### "Connection refused" error

ComfyUI is not running or not accessible:
```bash
# Check ComfyUI is running
curl http://127.0.0.1:8188/system_stats

# If not running, start ComfyUI first
```

### "Model not found" error

Check model paths in ComfyUI:
```bash
# Models should be in:
# ComfyUI/models/unet/Wan2.2/wan2.2_t2v_high_noise_14B_fp16.safetensors
# ComfyUI/models/unet/Wan2.2/wan2.2_t2v_low_noise_14B_fp16.safetensors
# ComfyUI/models/clip/umt5_xxl_fp16.safetensors
# ComfyUI/models/vae/wan_2.1_vae.safetensors
```

### Generation hangs or times out

Check Ray/GPU logs:
```bash
# In ComfyUI console, look for:
# - NCCL errors (communication issues)
# - OOM errors (out of memory)
# - Ray actor crashes
```

If NCCL issues occur:
```bash
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
# Restart ComfyUI
```

### Slow generation times

Expected times for reference (will vary based on system):
- **1280x720, 161 frames, 40 steps**: ~30-60s on 8× MI300X with USP
- **1920x1080, 161 frames, 40 steps**: ~60-120s
- **1920x1280, 241 frames, 50 steps**: ~120-240s

If significantly slower:
1. Check GPU utilization: `rocm-smi`
2. Verify Flash Attention is installed
3. Test different attention backends
4. Check for thermal throttling

## Advanced Options

### Modify Script for Different GPU Counts

Edit the workflow JSON:
```json
"38": {
  "inputs": {
    "GPU": 4,              // Change to your GPU count
    "ulysses_degree": 4,   // Should match GPU count
    // ...
  }
}
```

### Add Warm-up Runs

First run is often slower due to compilation. Add a warm-up:

```python
# In benchmark script, before main loop:
print("Running warm-up...")
warm_up_config = {
    'width': 1280, 'height': 720,
    'length': 81, 'steps': 20,
    'prompt': DEFAULT_PROMPT, 'seed': 0
}
run_benchmark(client, workflow_template, warm_up_config, 0)
print("Warm-up complete\n")
```

### Custom Prompts from File

```python
# Load prompts from file
with open('prompts.txt', 'r') as f:
    prompts = [line.strip() for line in f if line.strip()]

# Run benchmark with each prompt
for i, prompt in enumerate(prompts):
    config = {
        'width': 1280, 'height': 720,
        'length': 161, 'steps': 40,
        'prompt': prompt, 'seed': i
    }
    result = run_benchmark(client, workflow_template, config, i+1)
    results.append(result)
```

## Reference: Wan 2.2 Architecture

From the Hugging Face documentation:

> **Wan2.2 introduces Mixture-of-Experts (MoE) architecture** into the video generation diffusion model. The A14B model series adopts a two-expert design tailored to the denoising process:
>
> - **High-noise expert** for the early stages, focusing on overall layout
> - **Low-noise expert** for the later stages, refining video details
>
> Each expert model has about 14B parameters, resulting in a total of 27B parameters but only 14B active parameters per step.

The transition happens automatically based on the Signal-to-Noise Ratio (SNR) at timestep `t_moe` (typically around step 20 for 40 total steps).

## Citation

If you use this benchmark or raylight in your research:

```bibtex
@misc{raylight2025,
  title={Raylight: Multi-GPU Parallelism for ComfyUI Diffusion Models},
  year={2025},
  url={https://github.com/yourusername/raylight}
}

@article{wan2025,
  title={Wan: Open and Advanced Large-Scale Video Generative Models},
  author={Team Wan and contributors},
  journal={arXiv preprint arXiv:2503.20314},
  year={2025}
}
```

## Support

- Issues: https://github.com/yourusername/raylight/issues
- Discussions: https://github.com/yourusername/raylight/discussions
- Discord: [Your Discord Link]

---

**Happy Benchmarking!** 🚀

## FP8 Quantization Support

### Overview

The benchmark suite supports both **FP16** (standard precision) and **FP8** (quantized) models. FP8 quantization provides:

- **1.3-2.0x faster generation** (hardware dependent)
- **~50% reduction in model memory** (14GB → ~7GB per expert)
- **Minimal quality loss** (FP8 is well-suited for inference)
- **Excellent MI300X support** (AMD CDNA3 has strong FP8 Tensor Core support)

### Prerequisites

You need **both FP8 model files** downloaded:

```
ComfyUI/models/diffusion_models/Wan2.2/
├── wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors
└── wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors

ComfyUI/models/text_encoders/
└── umt5_xxl_fp8_e4m3fn_scaled.safetensors

ComfyUI/models/vae/
└── wan_2.1_vae.safetensors  # Same as FP16 version
```

**Download from**: [Comfy-Org/Wan_2.2_ComfyUI_Repackaged](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged)

### Usage

#### Quick FP8 Test

```bash
python benchmark_wan22_mi300x.py --quick --precision fp8
```

#### Full FP8 Benchmark

```bash
python benchmark_wan22_mi300x.py --precision fp8
```

This automatically:
- Loads the FP8 workflow template (`WanT2V_MI300X_FP8_Benchmark.json`)
- Uses FP8 model files
- Outputs results to `benchmark_results_fp8_TIMESTAMP.csv`

#### Compare FP16 vs FP8

Run both benchmarks and compare:

```bash
# Run FP16 benchmark
python benchmark_wan22_mi300x.py --quick --precision fp16 --output results_fp16.csv

# Run FP8 benchmark
python benchmark_wan22_mi300x.py --quick --precision fp8 --output results_fp8.csv

# Compare results
python -c "
import pandas as pd
fp16 = pd.read_csv('results_fp16.csv')
fp8 = pd.read_csv('results_fp8.csv')
speedup = fp16['generation_time'].values[0] / fp8['generation_time'].values[0]
print(f'FP16: {fp16[\"generation_time\"].values[0]:.2f}s')
print(f'FP8:  {fp8[\"generation_time\"].values[0]:.2f}s')
print(f'Speedup: {speedup:.2f}x')
"
```

### Workflow Differences

The FP8 workflow (`WanT2V_MI300X_FP8_Benchmark.json`) differs from FP16 in three nodes:

**Node 8 (Low-noise UNET):**
```json
{
  "unet_name": "Wan2.2/wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors",
  "weight_dtype": "fp8_e4m3fn"  // Changed from "default"
}
```

**Node 45 (High-noise UNET):**
```json
{
  "unet_name": "Wan2.2/wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors",
  "weight_dtype": "fp8_e4m3fn"  // Changed from "default"
}
```

**Node 15 (CLIP):**
```json
{
  "clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors"  // Changed from fp16
}
```

All other nodes remain identical (RayInitializer, samplers, VAE, etc.).

### Expected Performance

Based on typical FP8 characteristics on MI300X:

| Metric | FP16 | FP8 | Improvement |
|--------|------|-----|-------------|
| Speed | Baseline | 1.3-2.0x faster | ✅ |
| Memory | ~14GB/expert | ~7GB/expert | ✅ 50% reduction |
| Quality | Baseline | Minimal loss | ✅ |
| Compatibility | Full | Full | ✅ |

**Real-world example** (estimated for 1280x720, 161 frames, 40 steps):
- FP16: ~40s
- FP8: ~25s (1.6x speedup)

### When to Use FP8

**Use FP8 when:**
- ✅ Running on MI300X (excellent FP8 Tensor Core support)
- ✅ Maximizing throughput (batch processing)
- ✅ Memory constrained (can fit larger batches)
- ✅ Quality is acceptable (minimal degradation)

**Stick with FP16 when:**
- ⚠️ Absolute maximum quality required
- ⚠️ Testing/debugging (easier to compare with reference implementations)
- ⚠️ Hardware has poor FP8 support (older GPUs)

### Troubleshooting FP8

**"Model not found" errors:**
```bash
# Verify FP8 models exist
ls ComfyUI/models/diffusion_models/Wan2.2/*fp8*
ls ComfyUI/models/text_encoders/umt5_xxl_fp8*
```

**Slower than expected:**
- Check that Flash Attention is installed (FP8 benefits from it)
- Verify MI300X drivers are up-to-date (ROCm 6.4+)
- Try different attention backends (`SAGE_FP8` likely best)

**Quality issues:**
- FP8 should have minimal quality degradation
- If unacceptable, fall back to FP16
- Quality is model author's responsibility (you're just benchmarking)

### Technical Details

**FP8 Format**: `fp8_e4m3fn` (E4M3 - 1 sign bit, 4 exponent bits, 3 mantissa bits)
- Range: ±448
- Precision: ~1% relative error
- Well-suited for neural network activations and weights

**Raylight Implementation**:
- Uses `weight_dtype: "fp8_e4m3fn"` in RayUNETLoader
- Automatically handled by ComfyUI's weight loading system
- No code changes needed - just different model files

### CSV Output Format

Results include precision information:

```csv
run_id,timestamp,precision,width,height,length,steps,generation_time,...
1,2025-10-14T12:00:00,fp8,1280,720,161,40,25.3,...
2,2025-10-14T12:05:00,fp8,1920,1080,161,40,58.7,...
```

This allows easy filtering and comparison in analysis tools.

---

**Note**: FP8 performance varies by hardware. MI300X has excellent FP8 support via CDNA3 architecture. Actual speedup will depend on your specific configuration, attention backend, and workload.
