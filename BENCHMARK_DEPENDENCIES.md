# Benchmark Suite Dependencies

## Core Requirements

**None!** The benchmark scripts use only Python standard library.

The benchmark suite works with any **Python 3.8+** installation. No additional packages are required to run:
- `benchmark_wan22_mi300x.py`
- `benchmark_attention_backends.py`

## System Requirements

- **Python**: 3.8 or newer
- **ComfyUI**: Running instance with raylight installed
- **Models**: Wan 2.2 models (FP16 and/or FP8) downloaded to ComfyUI

## Optional Dependencies

For **result analysis and visualization**, you can install optional packages:

```bash
pip install -r requirements-benchmark.txt
```

This includes:
- **pandas** - CSV data analysis
- **matplotlib** - Performance plotting
- **jupyter** - Interactive notebooks
- **scipy/numpy** - Statistical analysis

## Quick Start (No Installation Needed)

```bash
# Works immediately with any Python 3.8+
python benchmark_wan22_mi300x.py --quick
```

## Result Analysis Examples

### Without Optional Packages (CSV only)

```bash
# Generate results
python benchmark_wan22_mi300x.py --quick --output results.csv

# View with standard tools
cat results.csv
head results.csv
```

### With Optional Packages (pandas)

```python
import pandas as pd

# Load and analyze
df = pd.read_csv('results.csv')
print(df.describe())
print(df.groupby('precision')['generation_time'].mean())
```

### With Visualization (matplotlib)

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('results.csv')
df.plot(x='length', y='generation_time', kind='scatter')
plt.xlabel('Frame Count')
plt.ylabel('Generation Time (s)')
plt.show()
```

## Why No Required Dependencies?

The benchmark scripts are designed to be **universally compatible**:

1. **No installation friction** - Works on any system with Python
2. **No version conflicts** - Doesn't interfere with ComfyUI's dependencies
3. **Portable** - Easy to copy to servers, containers, or air-gapped systems
4. **Reliable** - Standard library is always available and tested

## ComfyUI Integration

The benchmark scripts communicate with ComfyUI via HTTP API using only:
- `urllib.request` (HTTP client)
- `json` (JSON parsing)

No ComfyUI packages need to be imported - the scripts work as standalone clients.

## Docker/Container Usage

Since there are no dependencies, the scripts work in minimal containers:

```dockerfile
FROM python:3.8-slim
COPY benchmark_wan22_mi300x.py .
# That's it! No pip install needed
```

## Troubleshooting

**"ModuleNotFoundError: pandas"**
- This is expected if you haven't installed optional packages
- The benchmark scripts themselves don't need pandas
- Only install if you want to use the analysis examples in documentation

**"Python version too old"**
- Requires Python 3.8+
- Check: `python --version`
- Upgrade if needed

**"Connection refused"**
- ComfyUI not running, not a dependency issue
- Start ComfyUI first: `cd ComfyUI && python main.py`

---

**Summary**: The benchmark suite has **zero required dependencies** and works with any Python 3.8+ installation. Optional packages are only for post-processing results.
