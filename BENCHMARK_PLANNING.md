# Benchmark Script Enhancement Planning

## Overview
Planning document for creating a generic, workflow-agnostic ComfyUI benchmark script that can work with any Wan 2.2 (or other model) workflows without hardcoded node IDs.

## Current Limitations of `benchmark_wan22_mi300x.py`

### Hardcoded Node IDs (lines 266-302)
The `create_benchmark_workflow()` function assumes specific node structure:
- **Node 19**: `EmptyHunyuanLatentVideo` - sets width, height, length
- **Node 46**: High-noise sampler - sets steps, start_at_step, end_at_step, noise_seed
- **Node 14**: Low-noise sampler - sets steps, start_at_step, end_at_step
- **Node 17**: Text prompt input - sets text

This makes the script work ONLY with workflows that have these exact node IDs and types.

### Raylight-Specific References
- Default workflow paths point to Raylight workflows (lines 634-636)
- Error messages mention "Ray actor initialization" (line 704)
- Documentation assumes MI300X + Raylight setup

## Option 2: Auto-Detecting Workflow Structure

### Strategy: Node Type Detection

Instead of hardcoding node IDs, scan the workflow JSON to find nodes by their `class_type`:

```python
def find_nodes_by_type(workflow: Dict[str, Any], class_type: str) -> List[str]:
    """Find all node IDs matching a specific class_type"""
    return [node_id for node_id, node in workflow.items()
            if node.get('class_type') == class_type]

def create_benchmark_workflow_auto(
    template: Dict[str, Any],
    width: int,
    height: int,
    length: int,
    steps: int,
    prompt: str,
    seed: int
) -> Dict[str, Any]:
    """Create workflow with auto-detected node structure"""
    workflow = json.loads(json.dumps(template))  # Deep copy

    # Find nodes by type
    latent_nodes = find_nodes_by_type(workflow, 'EmptyHunyuanLatentVideo')
    sampler_nodes = find_nodes_by_type(workflow, 'KSamplerAdvanced')  # or specific sampler type
    prompt_nodes = find_nodes_by_type(workflow, 'CLIPTextEncode')  # or text input type

    # Update resolution/length on first latent node
    if latent_nodes:
        workflow[latent_nodes[0]]["inputs"]["width"] = width
        workflow[latent_nodes[0]]["inputs"]["height"] = height
        workflow[latent_nodes[0]]["inputs"]["length"] = length

    # Update samplers (logic for high/low noise split)
    if len(sampler_nodes) >= 2:
        high_noise_steps = steps // 2
        # Node 0: high noise
        workflow[sampler_nodes[0]]["inputs"]["steps"] = steps
        workflow[sampler_nodes[0]]["inputs"]["start_at_step"] = 0
        workflow[sampler_nodes[0]]["inputs"]["end_at_step"] = high_noise_steps
        workflow[sampler_nodes[0]]["inputs"]["noise_seed"] = seed

        # Node 1: low noise
        workflow[sampler_nodes[1]]["inputs"]["steps"] = steps
        workflow[sampler_nodes[1]]["inputs"]["start_at_step"] = high_noise_steps
        workflow[sampler_nodes[1]]["inputs"]["end_at_step"] = 10000
    elif len(sampler_nodes) == 1:
        # Single sampler workflow
        workflow[sampler_nodes[0]]["inputs"]["steps"] = steps
        workflow[sampler_nodes[0]]["inputs"]["seed"] = seed

    # Update prompt
    if prompt_nodes:
        workflow[prompt_nodes[0]]["inputs"]["text"] = prompt

    return workflow
```

### Node Type Mapping for Different Workflows

| Model | Latent Node Type | Sampler Node Type | Prompt Node Type |
|-------|------------------|-------------------|------------------|
| Wan 2.2 (Hunyuan) | `EmptyHunyuanLatentVideo` | `KSamplerAdvanced` | `CLIPTextEncode` |
| Flux | `EmptyLatentImage` | `KSampler` or `KSamplerAdvanced` | `CLIPTextEncode` |
| Qwen | TBD | TBD | TBD |
| Generic | Auto-detect | Auto-detect | Auto-detect |

### Configuration File Approach

Alternative: Use a YAML/JSON config to specify node mappings:

```yaml
# workflow_config.yaml
workflow_name: "Wan 2.2 Regular"
node_mapping:
  latent:
    node_id: "19"
    class_type: "EmptyHunyuanLatentVideo"
    params:
      width: "width"
      height: "height"
      length: "length"

  high_noise_sampler:
    node_id: "46"
    class_type: "KSamplerAdvanced"
    params:
      steps: "steps"
      start_step: "start_at_step"
      end_step: "end_at_step"
      seed: "noise_seed"

  low_noise_sampler:
    node_id: "14"
    class_type: "KSamplerAdvanced"
    params:
      steps: "steps"
      start_step: "start_at_step"
      end_step: "end_at_step"

  prompt:
    node_id: "17"
    class_type: "CLIPTextEncode"
    params:
      text: "text"

benchmark_configs:
  resolutions:
    - {width: 640, height: 480, name: "480P"}
    - {width: 1280, height: 720, name: "720P"}
    - {width: 1920, height: 1080, name: "1080P"}

  frame_counts: [81, 161, 241]
  step_counts: [20, 40, 50]
```

Then load this config instead of hardcoding:

```python
import yaml

def load_workflow_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_benchmark_workflow_from_config(
    template: Dict[str, Any],
    config: Dict[str, Any],
    width: int, height: int, length: int, steps: int, prompt: str, seed: int
) -> Dict[str, Any]:
    """Create workflow using config-specified node mappings"""
    workflow = json.loads(json.dumps(template))
    mapping = config['node_mapping']

    # Update latent node
    latent = mapping['latent']
    workflow[latent['node_id']]["inputs"][latent['params']['width']] = width
    workflow[latent['node_id']]["inputs"][latent['params']['height']] = height
    workflow[latent['node_id']]["inputs"][latent['params']['length']] = length

    # Update samplers...
    # (similar pattern for other nodes)

    return workflow
```

## Implementation Steps for New Script

### Phase 1: Core Script
1. Copy `benchmark_wan22_mi300x.py` to `benchmark_comfyui_generic.py`
2. Rename references (MI300X → Generic, Raylight → ComfyUI)
3. Update error messages to remove Raylight-specific mentions
4. Add `--node-config` argument for config file path

### Phase 2: Auto-Detection
1. Implement `find_nodes_by_type()` helper
2. Implement `create_benchmark_workflow_auto()` with type detection
3. Add fallback to manual config if auto-detection fails
4. Add `--detect-nodes` flag to print detected node structure

### Phase 3: Configuration System
1. Create YAML schema for workflow configs
2. Add example configs for common workflows:
   - `configs/wan22_regular.yaml`
   - `configs/wan22_raylight.yaml`
   - `configs/flux_dev.yaml`
3. Implement config loader and validator

### Phase 4: CLI Improvements
1. Add `--list-nodes` to inspect workflow structure
2. Add `--dry-run` to show what would be modified without running
3. Improve error messages when nodes not found
4. Add progress bar for multi-config runs

## Files to Create

```
raylight/
├── benchmark_comfyui_generic.py          # New generic script
├── benchmark_wan22_mi300x.py             # Keep existing (Raylight-optimized)
├── configs/
│   ├── wan22_regular.yaml                # Regular Wan 2.2 workflow config
│   ├── wan22_raylight_usp.yaml           # Raylight USP mode
│   ├── wan22_raylight_fsdp.yaml          # Raylight FSDP mode
│   └── flux_dev_regular.yaml             # Flux Dev workflow config
└── BENCHMARK_PLANNING.md                 # This file
```

## Example Usage (Future Script)

```bash
# Auto-detect workflow structure (simplest)
python benchmark_comfyui_generic.py --workflow my_workflow.json --auto-detect

# Use explicit config file
python benchmark_comfyui_generic.py --workflow my_workflow.json --config configs/wan22_regular.yaml

# Inspect workflow before running
python benchmark_comfyui_generic.py --workflow my_workflow.json --list-nodes

# Dry run to see what would be modified
python benchmark_comfyui_generic.py --workflow my_workflow.json --dry-run --quick

# Full benchmark with warmup
python benchmark_comfyui_generic.py --workflow my_workflow.json --config configs/wan22_regular.yaml --warmup 2
```

## Benefits of Option 2

1. **No hardcoded node IDs** - works with any workflow structure
2. **Reusable configs** - share configs for common workflow types
3. **Better error messages** - clearly indicates missing/incompatible nodes
4. **Workflow inspection** - helps users understand their workflow structure
5. **Future-proof** - easy to add new model types

## Current Workaround (Option 1)

For immediate use with regular workflows:

1. Export your workflow in API format from ComfyUI
2. Identify the node IDs for:
   - Latent/empty image node
   - Sampler node(s)
   - Text prompt node
3. Manually edit `create_benchmark_workflow()` function (lines 266-302) to use your node IDs
4. Run: `python benchmark_wan22_mi300x.py --workflow your_workflow.json --warmup 1`

## Identifying Node IDs in Your Workflow

```bash
# Pretty-print workflow to find node IDs
python -m json.tool your_workflow.json | grep -B 3 "class_type"

# Or use jq if installed
jq 'to_entries[] | {id: .key, type: .value.class_type}' your_workflow.json
```

Look for output like:
```json
{
  "id": "19",
  "type": "EmptyHunyuanLatentVideo"
}
{
  "id": "46",
  "type": "KSamplerAdvanced"
}
```

Then update the script with your actual node IDs.

## Decision Matrix

| Approach | Effort | Flexibility | Maintenance |
|----------|--------|-------------|-------------|
| **Option 1** (Current script) | Low (just update node IDs) | Low (workflow-specific) | High (manual edits) |
| **Option 2** (Auto-detect) | Medium (implement detection) | High (works with any workflow) | Low (self-adapting) |
| **Option 2** (Config-based) | High (implement + create configs) | Highest (shareable configs) | Medium (maintain configs) |

## Recommendation

- **Short term**: Use Option 1 - manually update node IDs for your specific workflow
- **Long term**: Implement Option 2 with auto-detection + config fallback for maximum flexibility

This allows immediate benchmarking while building toward a more robust solution.
