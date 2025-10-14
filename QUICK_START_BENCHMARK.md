# Quick Start: Benchmarking Wan 2.2 on MI300X

## TL;DR

```bash
# 1. Start ComfyUI with raylight
cd /path/to/ComfyUI
python main.py

# 2. In another terminal, run quick test (FP16)
cd /path/to/raylight
python benchmark_wan22_mi300x.py --quick

# 3. Run quick test with FP8 quantization (faster!)
python benchmark_wan22_mi300x.py --quick --precision fp8

# 4. Run full benchmark
python benchmark_wan22_mi300x.py --precision fp8

# 5. Compare attention backends
python benchmark_attention_backends.py
```

## What You Get

### 1. Quick Test (~5 minutes)
- Tests 1 configuration: 1280x720, 161 frames, 40 steps
- Validates your setup works
- Gives you a baseline number

### 2. Full Benchmark (~2-3 hours)
- Tests 27 configurations
- 3 resolutions × 3 frame counts × 3 step counts
- Comprehensive performance profile
- CSV output for analysis

### 3. Attention Backend Comparison (~30 minutes)
- Tests all 5 attention backends
- Finds fastest implementation for MI300X
- Compares: FLASH_ATTN, SAGE_FP8, SAGE_FP16_CUDA, SAGE_FP16_TRITON, TORCH

## Files Created

- `benchmark_wan22_mi300x.py` - Main benchmark script (supports FP16 and FP8)
- `benchmark_attention_backends.py` - Backend comparison script
- `example_workflows/WanT2V_MI300X_Throughput_Benchmark.json` - FP16 workflow template
- `example_workflows/WanT2V_MI300X_FP8_Benchmark.json` - FP8 workflow template
- `BENCHMARK_README.md` - Full documentation
- `benchmark_results_*.csv` - Results (auto-generated)

## Expected Results (Estimate)

On 8× MI300X with USP (sequence parallel):

### FP16 (Standard Precision)
| Resolution | Frames | Steps | Est. Time |
|------------|--------|-------|-----------|
| 1280x720   | 81     | 20    | ~15-25s   |
| 1280x720   | 161    | 40    | ~30-50s   |
| 1920x1080  | 161    | 50    | ~60-100s  |
| 1920x1280  | 241    | 50    | ~120-200s |

### FP8 (Quantized - Expected 1.3-2.0x Faster)
| Resolution | Frames | Steps | Est. Time |
|------------|--------|-------|-----------|
| 1280x720   | 81     | 20    | ~10-18s   |
| 1280x720   | 161    | 40    | ~20-35s   |
| 1920x1080  | 161    | 50    | ~40-70s   |
| 1920x1280  | 241    | 50    | ~80-140s  |

**Note**: These are rough estimates. Actual times depend on:
- GPU interconnect (PCIe vs Infinity Fabric)
- Attention backend used
- Model loading overhead
- System configuration
- **MI300X has excellent FP8 support** - expect good speedups!

## Next Steps

1. **Run quick test** to validate setup
2. **Review BENCHMARK_README.md** for detailed docs
3. **Run full benchmark** overnight
4. **Test attention backends** to find optimal config
5. **Share results** with the community!

## Important Notes

### About Wan 2.2 Models

You **MUST use BOTH** high-noise and low-noise models:

**FP16 Models (default):**
- `wan2.2_t2v_high_noise_14B_fp16.safetensors`
- `wan2.2_t2v_low_noise_14B_fp16.safetensors`
- `umt5_xxl_fp16.safetensors` (CLIP encoder)

**FP8 Models (quantized - faster, lower memory):**
- `wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors`
- `wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors`
- `umt5_xxl_fp8_e4m3fn_scaled.safetensors` (CLIP encoder)

**Both models are required** - this is not optional - it's how Wan 2.2 works (MoE architecture).

**Model Locations:**
```
ComfyUI/models/diffusion_models/Wan2.2/wan2.2_t2v_*_14B_*.safetensors
ComfyUI/models/text_encoders/umt5_xxl_*.safetensors
ComfyUI/models/vae/wan_2.1_vae.safetensors  # Same for both FP16 and FP8
```

### GPU Configuration

Your workflow is set for **8 GPUs**. If you have different count:

Edit `example_workflows/WanT2V_MI300X_Throughput_Benchmark.json`:
```json
"38": {
  "inputs": {
    "GPU": 8,              // <- Change this
    "ulysses_degree": 8,   // <- And this (must match GPU count)
    ...
  }
}
```

### FSDP Note

**FSDP is disabled** in the workflow - this is correct for MI300X:
- 192GB VRAM per GPU = no need for weight sharding
- FSDP adds communication overhead
- Use USP (sequence parallel) instead for latency

Only enable FSDP if you run out of VRAM.

## Troubleshooting

**"Connection refused"** → ComfyUI not running  
**"Model not found"** → Check model paths in ComfyUI/models/  
**"Timeout"** → Check Ray/NCCL logs in ComfyUI console  
**Very slow** → Check GPU utilization with `rocm-smi`, try different attention backend  

See `BENCHMARK_README.md` for detailed troubleshooting.

---

**Happy Benchmarking!** If you get good results, please share them with the community! 🚀
