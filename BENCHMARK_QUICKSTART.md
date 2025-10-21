# Wan 2.2 Benchmark Quick Start Guide

## Prerequisites

```bash
# 1. Clone Wan 2.2 repository
git clone https://github.com/Wan-Video/Wan2.2.git
cd Wan2.2
pip install -r requirements.txt

# 2. Download models (choose based on your needs)
huggingface-cli download Wan-AI/Wan2.2-T2V-A14B --local-dir ./Wan2.2-T2V-A14B
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir ./Wan2.2-TI2V-5B
```

## Quick Benchmark Commands

### AMD MI300X (8 GPUs, 192GB each)

#### Quick Test - T2V Model
```bash
# Using the wrapper script
python benchmark_wan22_native.py \
  --wan-repo ./Wan2.2 \
  --task t2v-A14B \
  --ckpt-dir ./Wan2.2-T2V-A14B \
  --num-gpus 8 \
  --quick \
  --warmup 1

# Direct torchrun command
torchrun --nproc_per_node=8 generate.py \
  --task t2v-A14B \
  --size 640*480 \
  --ckpt_dir ./Wan2.2-T2V-A14B \
  --dit_fsdp \
  --t5_fsdp \
  --ulysses_size 8 \
  --inference_steps 20 \
  --prompt "Test video generation"
```

#### Full Benchmark Matrix
```bash
# Full test suite (27 configurations)
python benchmark_wan22_native.py \
  --wan-repo ./Wan2.2 \
  --task t2v-A14B \
  --ckpt-dir ./Wan2.2-T2V-A14B \
  --num-gpus 8 \
  --warmup 2 \
  --output benchmark_mi300x_8gpu.csv
```

### NVIDIA H100 (8 GPUs, 80GB each)

```bash
# Enable TF32 and Flash Attention 3
export TORCH_ALLOW_TF32_CUBLAS_GEMM=1
export FLASH_ATTENTION_SKIP_GEMM=0

# Run benchmark
torchrun --nproc_per_node=8 generate.py \
  --task t2v-A14B \
  --size 1280*720 \
  --ckpt_dir ./Wan2.2-T2V-A14B \
  --dit_fsdp \
  --t5_fsdp \
  --ulysses_size 8 \
  --inference_steps 40 \
  --prompt "Epic AI data center visualization"
```

### Consumer GPU (RTX 4090, 24GB)

```bash
# TI2V-5B model optimized for consumer GPUs
python benchmark_wan22_native.py \
  --wan-repo ./Wan2.2 \
  --task ti2v-5B \
  --ckpt-dir ./Wan2.2-TI2V-5B \
  --quick \
  --warmup 1

# Direct command with memory optimizations
python generate.py \
  --task ti2v-5B \
  --size 704*1280 \
  --ckpt_dir ./Wan2.2-TI2V-5B \
  --offload_model True \
  --convert_model_dtype \
  --t5_cpu \
  --inference_steps 20 \
  --prompt "Test generation"
```

## Benchmark Configurations

| Resolution | Frames | Steps | Description |
|------------|--------|-------|-------------|
| 640×480 (480P) | 81, 161, 241 | 20, 40, 50 | Low res, fast testing |
| 1280×720 (720P) | 81, 161, 241 | 20, 40, 50 | Standard HD |
| 1920×1080 (1080P) | 81, 161, 241 | 20, 40, 50 | Full HD |

### Frame Counts (24fps)
- **81 frames**: ~3.4 seconds
- **161 frames**: ~6.7 seconds
- **241 frames**: ~10 seconds

### Step Counts
- **20 steps**: Fast preview quality
- **40 steps**: Production quality
- **50 steps**: High quality

## Custom Configurations

### Specific Test Case
```bash
# 1080P, 161 frames, 50 steps on 8 GPUs
python benchmark_wan22_native.py \
  --wan-repo ./Wan2.2 \
  --task t2v-A14B \
  --ckpt-dir ./Wan2.2-T2V-A14B \
  --num-gpus 8 \
  --width 1920 \
  --height 1080 \
  --length 161 \
  --steps 50 \
  --warmup 1
```

### Custom Prompt
```bash
python benchmark_wan22_native.py \
  --wan-repo ./Wan2.2 \
  --task t2v-A14B \
  --ckpt-dir ./Wan2.2-T2V-A14B \
  --num-gpus 8 \
  --quick \
  --prompt "A serene mountain landscape with flowing waterfalls and birds flying" \
  --warmup 1
```

## Environment Setup

### AMD MI300X
```bash
export HSA_FORCE_FINE_GRAIN_PCIE=1
export ROCM_PATH=/opt/rocm
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# If NCCL timeouts occur
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_TIMEOUT=1800
```

### NVIDIA H100/A100
```bash
export TORCH_ALLOW_TF32_CUBLAS_GEMM=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# For debugging
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
```

## Output Analysis

### View Results
```bash
# Quick summary
cat benchmark_mi300x_8gpu.csv | column -t -s,

# Python analysis
python -c "
import pandas as pd
df = pd.read_csv('benchmark_mi300x_8gpu.csv')
print('Average generation time by resolution:')
print(df.groupby(['width', 'height'])['generation_time'].mean())
print('\nAverage FPS:')
print(df['frames_per_second'].mean())
"
```

### Expected Performance

| GPU Config | Resolution | Frames | Steps | Expected Time |
|------------|------------|--------|-------|---------------|
| 8× MI300X | 720P | 161 | 40 | ~45-60s |
| 8× MI300X | 1080P | 161 | 40 | ~90-120s |
| 8× H100 | 720P | 161 | 40 | ~40-55s |
| 1× RTX 4090 | 720P | 120 | 20 | ~540s (9 min) |

## Troubleshooting

### Out of Memory
```bash
# Add memory optimizations
--offload_model True \
--convert_model_dtype \
--t5_cpu
```

### NCCL Timeout
```bash
export NCCL_TIMEOUT=3600  # 1 hour
export NCCL_P2P_DISABLE=1
```

### Slow First Run
- Normal - model loading and kernel compilation
- Use `--warmup 2` for accurate timing

## Comparison: Native vs ComfyUI

| Metric | Native Wan 2.2 | ComfyUI Workflow |
|--------|----------------|------------------|
| Setup Complexity | Simple | Requires server |
| Overhead | Minimal | Queue + node processing |
| Memory Usage | Model only | ComfyUI + model |
| Flexibility | Script-based | Visual workflow |
| Best For | Benchmarking | Production workflows |

## Complete Example Workflow

```bash
# 1. Setup environment
cd Wan2.2
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 2. Run warmup
torchrun --nproc_per_node=8 generate.py \
  --task t2v-A14B \
  --size 1280*720 \
  --ckpt_dir ./Wan2.2-T2V-A14B \
  --dit_fsdp --t5_fsdp --ulysses_size 8 \
  --inference_steps 20 \
  --num_clip 1 \
  --prompt "Warmup test"

# 3. Run benchmark suite
python ../benchmark_wan22_native.py \
  --wan-repo . \
  --task t2v-A14B \
  --ckpt-dir ./Wan2.2-T2V-A14B \
  --num-gpus 8 \
  --warmup 0 \
  --output results.csv

# 4. Analyze results
python -c "
import pandas as pd
df = pd.read_csv('results.csv')
print(df.describe())
"
```