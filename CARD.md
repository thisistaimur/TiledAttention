---
library_name: kernels
{% if license %}license: {{ license }}
{% endif %}---

TiledAttention is a scaled dot-product attention (SDPA) forward kernel for NVIDIA GPUs, implemented in cuTile Python (TileIR) and exposed for PyTorch-oriented workflows. The design follows FlashAttention-style online softmax with tiled (K,V) streaming, while emphasizing schedule-level modifiability (tile shapes, staging, shared-memory layout) for reproducible kernel research.

In the accompanying study, TiledAttention is evaluated against PyTorch SDPA auto-dispatch and explicit baselines across sequence length, head dimension, causal/non-causal masking, and FP16/BF16 precision.

[![arXiv](https://img.shields.io/badge/arXiv-2603.01960-b31b1b.svg)](https://doi.org/10.48550/arXiv.2603.01960)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18787737.svg)](https://doi.org/10.5281/zenodo.18787737)

## How to use
{% if functions %}

```python
# make sure `kernels` is installed: `pip install -U kernels`
from kernels import get_kernel

kernel_module = get_kernel("{{ repo_id }}", version={{ version }})
{{ functions[0] }} = kernel_module.{{ functions[0] }}

{{ functions[0] }}(...)
```
{% else %}

Usage example not available.
{% endif %}

## Available functions
{% if functions %}
{% for func in functions %}
- `{{ func }}`
{% endfor %}
{% else %}

Function list not available.
{% endif %}
{% if layers %}

## Available layers
{% for layer in layers %}
- `{{ layer }}`
{% endfor %}
{% endif %}

## Benchmarks
{% if has_benchmark %}

Benchmarking script is available for this kernel. Run `kernels benchmark {{ repo_id }} --version {{ version }}`.
{% else %}

No benchmark available yet.
{% endif %}
{% if upstream %}

## Source code

Source code of this kernel originally comes from {{ upstream }} and it was repurposed for compatibility with `kernels`.
{% endif %}
