# QuantStream MLX

QuantStream MLX is a memory-aware inference engine designed to run quantized language models larger than the available unified memory on Apple Silicon.

It reduces memory usage by combining:

- INT4 and INT8 weight quantization
- Layer-by-layer model streaming
- Adaptive layer caching
- Asynchronous layer prefetching
- MLX-based computation

> This project is currently under development.

## Problem

Large language models contain billions of weights distributed across multiple transformer layers.

A model normally loads all its layers into memory:

```text
Input
  ↓
Embedding Layer
  ↓
Transformer Layer 1
  ↓
Transformer Layer 2
  ↓
...
  ↓
Transformer Layer N
  ↓
Output Head
  ↓
Next Token
```

If a model requires 64 GB of memory but the computer has only 16 GB, the complete model cannot normally remain in memory.

QuantStream stores all quantized layers on the SSD but loads only a limited number of layers into unified memory at a time.

## Important Concept

Each transformer layer has its own weights:

```text
Layer 1 → Weights W1
Layer 2 → Weights W2
Layer 3 → Weights W3
```

The weights do not move between layers.

The data passed from one layer to another is called the hidden state:

```text
Input x
   ↓
Layer 1 uses W1
   ↓
Hidden State h1
   ↓
Layer 2 uses W2
   ↓
Hidden State h2
   ↓
Layer 3 uses W3
   ↓
Output
```

During inference, the weights remain unchanged.

## Quantization

Quantization stores model weights using fewer bits.

| Format | Bits per weight | Approximate model size |
|---|---:|---:|
| FP16 | 16 bits | 64 GB |
| INT8 | 8 bits | 32 GB |
| INT4 | 4 bits | 16–18 GB |

Quantization does not reduce the number of weights. It reduces the amount of memory required to store each weight.

## Quantization Example

Consider a group of floating-point weights:

```python
weights = [-0.210, -0.078, 0.026, 0.130, 0.182]
```

For symmetric INT4 quantization, values are represented approximately between `-7` and `7`.

```python
qmax = 7

max_value = max(abs(weight) for weight in weights)
scale = max_value / qmax

quantized_weights = [
    round(weight / scale)
    for weight in weights
]
```

The result is:

```text
Scale: 0.030

Original weights:
[-0.210, -0.078, 0.026, 0.130, 0.182]

Quantized weights:
[-7, -3, 1, 4, 6]
```

The approximate original weights can be reconstructed using:

```python
reconstructed_weights = [
    quantized_weight * scale
    for quantized_weight in quantized_weights
]
```

A single scale is shared by a group of weights. Every weight does not require its own scale.

The final implementation will use MLX quantized layers and low-bit computation instead of reconstructing the complete model in FP16.

## Layer Streaming

All quantized layers are stored on the SSD:

```text
model/
├── embeddings.safetensors
├── layer_000.safetensors
├── layer_001.safetensors
├── layer_002.safetensors
├── ...
├── final_norm.safetensors
└── output_head.safetensors
```

During inference, QuantStream performs the following operations:

```text
Load Layer 1
      ↓
Process Hidden State
      ↓
Unload Layer 1
      ↓
Load Layer 2
      ↓
Process Hidden State
      ↓
Unload Layer 2
      ↓
Continue through every layer
      ↓
Generate Next Token
```

All layer weights remain stored on the SSD, but only the currently required layers remain in memory.

## Adaptive Caching

Loading every layer from the SSD for every generated token can be slow.

The adaptive cache keeps as many layers in unified memory as the configured memory budget allows.

```text
Available Memory
      ↓
Reserve space for hidden states and KV cache
      ↓
Use remaining space for model layers
      ↓
Keep some layers resident
      ↓
Stream the remaining layers
```

As the KV cache grows, the scheduler can reduce the number of cached layers.

## Asynchronous Prefetching

While MLX processes the current layer group, QuantStream loads the next group in the background.

```text
Time ──────────────────────────────────────>

Compute:  [Layers 0–1][Layers 2–3][Layers 4–5]
Loading:        [Load 2–3][Load 4–5][Load 6–7]
```

The goal is to overlap SSD loading with model computation and reduce waiting time.

## System Architecture

```text
Quantized Layers on SSD
            ↓
        Layer Store
            ↓
    Memory-Aware Scheduler
       ↙             ↘
Layer Cache       Prefetcher
       ↘             ↙
      Apple Unified Memory
                ↓
           MLX Computation
                ↓
           Hidden State
                ↓
          Next Model Layer
```

## Project Structure

```text
quantstream-mlx/
├── README.md
├── pyproject.toml
├── src/
│   └── quantstream/
│       ├── __init__.py
│       ├── config.py
│       ├── engine.py
│       ├── model_adapter.py
│       ├── quantizer.py
│       ├── layer_store.py
│       ├── scheduler.py
│       ├── cache.py
│       ├── prefetcher.py
│       └── profiler.py
├── scripts/
│   ├── prepare_model.py
│   ├── generate.py
│   └── benchmark.py
└── tests/
    ├── test_quantization.py
    ├── test_layer_outputs.py
    ├── test_generation.py
    └── test_memory_budget.py
```

## Installation

QuantStream requires an Apple Silicon Mac.

```bash
git clone https://github.com/your-username/quantstream-mlx.git
cd quantstream-mlx

python3 -m venv .venv
source .venv/bin/activate

pip install -e .
```

## Dependencies

```toml
[project]
name = "quantstream-mlx"
version = "0.1.0"
requires-python = ">=3.11"

dependencies = [
    "mlx",
    "mlx-lm",
    "safetensors",
    "transformers",
    "psutil"
]
```

## Intended Usage

```python
from quantstream import QuantStream

engine = QuantStream.from_pretrained(
    model_path="models/tinyllama-streamed",
    quantization="int4",
    max_memory_gb=4.0,
    layers_per_chunk="auto",
    cache_policy="memory_aware",
    prefetch=True,
)

result = engine.generate(
    prompt="Explain virtual memory in simple terms.",
    max_new_tokens=64,
    temperature=0.0,
)

print(result.text)
print(result.metrics)
```

Example metrics:

```python
{
    "peak_memory_gb": 3.7,
    "tokens_per_second": 2.8,
    "layer_load_seconds": 5.1,
    "layer_compute_seconds": 3.4,
    "prefetch_wait_seconds": 0.8,
    "cache_hits": 18,
    "cache_misses": 26
}
```

These values are examples. Final results will be collected from actual experiments.

## Core Inference Logic

```python
def forward(input_ids, kv_cache):
    hidden_state = embedding_layer(input_ids)

    for layer_group in scheduler.layer_groups():
        layers = prefetcher.get(layer_group)

        next_group = scheduler.next_group(layer_group)

        if next_group is not None:
            prefetcher.submit(next_group)

        for layer_id, layer in layers:
            hidden_state, kv_cache[layer_id] = layer(
                hidden_state,
                kv_cache=kv_cache[layer_id],
            )

        cache.release_or_keep(layer_group, layers)

    hidden_state = final_norm(hidden_state)
    logits = output_head(hidden_state)

    return logits
```

This is architectural pseudocode. The implementation must also handle attention masks, rotary positional embeddings, MLX lazy evaluation and KV-cache management.

## Memory-Aware Scheduler

The scheduler calculates how much memory can be used for model layers:

```text
Layer memory budget =
    Total configured memory
    - Embedding memory
    - Output-head memory
    - Hidden-state memory
    - KV-cache memory
    - Safety margin
```

A basic chunk-size calculation:

```python
def choose_chunk_size(layer_sizes, available_bytes):
    selected_layers = []
    used_bytes = 0

    for layer_id, layer_size in enumerate(layer_sizes):
        if selected_layers and used_bytes + layer_size > available_bytes:
            break

        selected_layers.append(layer_id)
        used_bytes += layer_size

    if not selected_layers:
        raise MemoryError(
            "A single layer is larger than the available memory budget."
        )

    return len(selected_layers)
```

## Asynchronous Prefetcher

```python
from concurrent.futures import ThreadPoolExecutor


class Prefetcher:
    def __init__(self, layer_store):
        self.layer_store = layer_store
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.pending = {}

    def submit(self, layer_ids):
        key = tuple(layer_ids)

        if key not in self.pending:
            self.pending[key] = self.executor.submit(
                self._load_layers,
                layer_ids,
            )

    def get(self, layer_ids):
        key = tuple(layer_ids)

        if key not in self.pending:
            self.submit(layer_ids)

        return self.pending.pop(key).result()

    def _load_layers(self, layer_ids):
        return [
            (layer_id, self.layer_store.load(layer_id))
            for layer_id in layer_ids
        ]
```

## Correctness Testing

A generated response that looks reasonable is not enough to prove correctness.

The project compares:

1. Normal FP16 inference
2. Streamed FP16 inference
3. Streamed INT8 inference
4. Streamed INT4 inference
5. Streamed INT4 inference with caching and prefetching

FP16 streaming should closely match normal FP16 inference:

```python
mx.testing.assert_allclose(
    streamed_logits,
    baseline_logits,
    rtol=1e-3,
    atol=1e-3,
)
```

Quantized models will introduce rounding differences. Their quality will be evaluated using:

- Logit similarity
- Perplexity
- Deterministic token comparison
- Standard language-model evaluation datasets

## Benchmarking

| Configuration | Peak memory | Prefill tokens/s | Decode tokens/s | Quality |
|---|---:|---:|---:|---:|
| Standard MLX FP16 | TBD | TBD | TBD | Baseline |
| Streamed FP16 | TBD | TBD | TBD | TBD |
| Streamed INT8 | TBD | TBD | TBD | TBD |
| Streamed INT4 | TBD | TBD | TBD | TBD |
| INT4 with caching | TBD | TBD | TBD | TBD |
| INT4 with caching and prefetching | TBD | TBD | TBD | TBD |

Each benchmark will report:

- Mac model
- Apple Silicon chip
- Unified-memory capacity
- macOS version
- MLX version
- Model name and revision
- Quantization bit width
- Quantization group size
- Context length
- Generated-token count
- Peak memory
- Time to first token
- Tokens per second
- SSD bytes read per token

## Development Roadmap

### Phase 1: FP16 Layer Streaming

- Support one Llama-compatible model.
- Separate the model into independently loadable layers.
- Implement the layer-by-layer forward pass.
- Match the baseline model's hidden states and logits.

### Phase 2: Quantization

- Add INT8 quantization.
- Add INT4 quantization using MLX.
- Save quantized layers and scales.
- Measure memory reduction and output differences.

### Phase 3: Adaptive Caching

- Add a configurable memory budget.
- Keep as many layers resident as memory permits.
- Account for KV-cache growth.
- Dynamically evict layers when required.

### Phase 4: Asynchronous Prefetching

- Load upcoming layers using a background worker.
- Overlap SSD loading with MLX computation.
- Measure loading wait time.
- Optimize MLX evaluation boundaries.

### Phase 5: Benchmarking

- Compare against standard MLX-LM inference.
- Publish reproducible benchmark scripts.
- Test multiple memory limits and context lengths.
- Document performance and quality trade-offs.

## Technical Challenges

- SSD loading may make token generation slow.
- Every generated token must pass through every transformer layer.
- The KV cache grows with the context length.
- Apple Silicon uses unified memory, so RAM and VRAM are not separate pools.
- MLX uses lazy evaluation, which affects synchronization and prefetching.
- Dequantizing complete layers may create large temporary allocations.
- Quantized output may differ from the original model.

## Project Goal

The goal is to produce a measurable result such as:

> QuantStream executed a quantized language model under a restricted Apple unified-memory budget. Adaptive caching reduced SSD reads, while asynchronous prefetching reduced layer-loading wait time with a measured quality trade-off.

All performance numbers will be based on reproducible experiments.

## Non-Goals

The first version will not support:

- Model training
- Model fine-tuning
- CUDA GPUs
- Every Hugging Face architecture
- Distributed inference
- Guaranteed faster inference than an in-memory model

The MVP focuses on running a model under a strict memory limit and measuring the resulting speed, memory and quality trade-offs.

## License

This project will be released under the MIT License.

Model checkpoints remain subject to their respective licenses and usage conditions.
