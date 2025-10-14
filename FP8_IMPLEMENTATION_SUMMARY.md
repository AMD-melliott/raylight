# FP8 Quantization Support - Implementation Summary

## Overview

Successfully implemented FP8 quantization support for the Wan 2.2 benchmark suite, allowing performance comparison between FP16 (standard precision) and FP8 (quantized) models.

## Files Created/Modified

### New Files

1. **`example_workflows/WanT2V_MI300X_FP8_Benchmark.json`**
   - FP8-specific workflow template
   - Changes from FP16 version:
     - Node 8: `wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors` + `weight_dtype: "fp8_e4m3fn"`
     - Node 45: `wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors` + `weight_dtype: "fp8_e4m3fn"`
     - Node 15: `umt5_xxl_fp8_e4m3fn_scaled.safetensors`
     - Node 23: Output prefix changed to `"raylight/gen_fp8_"`

### Modified Files

2. **`benchmark_wan22_mi300x.py`**
   - Added `--precision` argument (`choices=['fp16', 'fp8']`, default='fp16')
   - Auto-selects workflow based on precision if `--workflow` not specified
   - Auto-generates output filename with precision tag
   - Updated `run_benchmark()` to accept `precision` parameter
   - Added `precision` field to result dictionary and CSV output
   - Enhanced examples in help text

3. **`QUICK_START_BENCHMARK.md`**
   - Updated TL;DR with FP8 examples
   - Added FP8 workflow to files list
   - Added FP8 performance estimates table
   - Documented FP8 model requirements and locations
   - Updated model paths section

4. **`BENCHMARK_README.md`**
   - Added comprehensive "FP8 Quantization Support" section
   - Includes: prerequisites, usage, comparison examples, technical details
   - Troubleshooting guide for FP8-specific issues
   - Performance expectations and recommendations

## Usage Examples

### Quick FP8 Test
```bash
python benchmark_wan22_mi300x.py --quick --precision fp8
```

### Full FP8 Benchmark
```bash
python benchmark_wan22_mi300x.py --precision fp8
```

### Compare FP16 vs FP8
```bash
# FP16
python benchmark_wan22_mi300x.py --quick --precision fp16 --output results_fp16.csv

# FP8
python benchmark_wan22_mi300x.py --quick --precision fp8 --output results_fp8.csv
```

## Key Features

1. **Automatic workflow selection** - `--precision fp8` automatically loads FP8 workflow
2. **Precision tracking** - All CSV results include `precision` column
3. **Auto-generated filenames** - Output files tagged with precision (e.g., `benchmark_results_fp8_20251014_120000.csv`)
4. **Backward compatible** - Default behavior unchanged (FP16), no breaking changes

## Technical Implementation

### FP8 Configuration

- **Format**: `fp8_e4m3fn` (E4M3 - 1 sign, 4 exponent, 3 mantissa bits)
- **Weight dtype**: Standard mode (not `fp8_e4m3fn_fast` per user preference)
- **Models required**:
  - `wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors`
  - `wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors`
  - `umt5_xxl_fp8_e4m3fn_scaled.safetensors`

### Workflow Changes

Only 3 nodes differ between FP16 and FP8 workflows:
- Node 8: Low-noise UNET (model path + weight_dtype)
- Node 45: High-noise UNET (model path + weight_dtype)
- Node 15: CLIP encoder (model path)

All other nodes identical (RayInitializer, samplers, VAE, etc.)

## Expected Performance

On AMD MI300X GPUs:

| Metric | Improvement |
|--------|-------------|
| Speed | 1.3-2.0x faster |
| Memory | ~50% reduction (14GB → 7GB per expert) |
| Quality | Minimal degradation |

**Example** (1280x720, 161 frames, 40 steps):
- FP16: ~40s
- FP8: ~25s (1.6x faster)

## CSV Output Format

Results now include precision information:

```csv
run_id,timestamp,precision,width,height,length,steps,generation_time,total_time,...
1,2025-10-14T12:00:00,fp8,1280,720,161,40,25.3,25.5,...
2,2025-10-14T12:05:00,fp16,1280,720,161,40,40.1,40.3,...
```

## Validation Checklist

- [x] FP8 workflow template created
- [x] `--precision` flag added to benchmark script
- [x] Automatic workflow selection based on precision
- [x] Precision field added to CSV output
- [x] Auto-generated filenames with precision tag
- [x] Documentation updated (QUICK_START and BENCHMARK_README)
- [x] Examples provided for all usage scenarios
- [x] Backward compatibility maintained

## Next Steps for User

1. Download FP8 models from Hugging Face
2. Place in correct ComfyUI model directories
3. Run quick test: `python benchmark_wan22_mi300x.py --quick --precision fp8`
4. Compare with FP16: `python benchmark_wan22_mi300x.py --quick --precision fp16`
5. Analyze speedup and decide optimal precision for production use

## Notes

- Implementation follows Option A (separate workflows + comparison capability)
- Uses standard FP8 (`fp8_e4m3fn`) not fast variant per user requirement
- Focus on performance benchmarking only (quality comparison out of scope)
- All changes are non-breaking - default behavior preserved

---

**Implementation Status**: ✅ Complete
**Date**: 2025-10-14
