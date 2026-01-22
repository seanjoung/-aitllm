# aitllm - Layer-Streaming LLM Inference Runtime

Production-grade inference engine for running 7B-8B parameter LLMs on low-VRAM GPUs (e.g., RTX 2070 8GB).

## Core Features

- **Layer-wise streaming**:  Weights never fully resident in GPU memory
- **CPU KV cache spill**:  Only current layer KV on GPU, rest on CPU pinned memory
- **Double buffering + prefetch**: Background loading while layer executes
- **INT4 on-the-fly dequant**: Compressed storage with runtime expansion
- **Multi-model support**: Qwen2.5-7B-Instruct, LLaMA 3.x (8B)

## Architecture

```
Disk (safetensors) → CPU Buffer A/B → GPU Layer Execute → Free
                         ↓
                    Prefetch Thread
                         ↓
                    KV Cache Tier (CPU pinned ↔ GPU active)
```

## Installation

```bash
# Prerequisites:  CUDA 12.x, PyTorch 2.1+
pip install -r requirements.txt
```

## Quick Start

```bash
# Run demo inference
python run_demo.py --model Qwen/Qwen2.5-7B-Instruct --prompt "Explain quantum computing"
```

## Usage

```python
from aitllm import StreamingEngine

engine = StreamingEngine(
    model_path="Qwen/Qwen2.5-7B-Instruct",
    max_length=512,
    device="cuda:0"
)

output = engine.generate("What is the meaning of life?", max_new_tokens=128)
print(output)
```

## Benchmarks

See `notebooks/03_tps_benchmark.ipynb` for detailed performance analysis.

**Target Performance (RTX 2070 8GB)**: 
- Qwen2.5-7B:  ~8-12 tokens/sec (FP16 layers, INT4 storage)
- LLaMA 3-8B: ~7-10 tokens/sec
- VRAM usage: <6GB peak

## Architecture Details

### Layer Streaming Pipeline

1. **Shard Storage** (`storage/shard.py`): Splits model into per-layer safetensors
2. **Prefetch Worker** (`storage/prefetch.py`): Background thread loads layer N+1
3. **Double Buffer** (`runtime/buffer.py`): Swaps between A/B buffers
4. **GPU Executor** (`model/block.py`): Runs single layer, frees immediately
5. **KV Manager** (`model/kv_cache.py`): Moves KV between CPU/GPU tiers

### Memory Guarantees

- **GPU**:  1 layer weights (~350MB) + 1 layer KV (~50MB/token) + activations (~200MB)
- **CPU**: Full KV cache (~1.5GB for 512 tokens) + 2 layer buffers (~700MB)
- **Disk**: INT4 compressed model (~4GB) or FP16 sharded (~14GB)

## Model Support

| Model | Status | Notes |
|-------|--------|-------|
| Qwen2.5-7B-Instruct | ✅ Full | Primary target |
| LLaMA 3-8B | ✅ Full | Uses RoPE adapter |
| Mistral 7B | 🚧 Partial | Sliding window needs work |

## Comparison to airllm

| Feature | airllm | aitllm |
|---------|--------|--------|
| Layer streaming | ✅ | ✅ |
| KV cache spill | ❌ (GPU OOM) | ✅ CPU tier |
| Prefetch | ❌ Sequential | ✅ Background thread |
| INT4 support | ❌ | ✅ On-the-fly |
| Model coverage | Limited | Extensible adapters |

## Development

```bash
# Create sharded model
python -m aitllm.storage.shard --model Qwen/Qwen2.5-7B-Instruct --output ./shards

# Run streaming test
jupyter notebook notebooks/02_streaming.ipynb

# Benchmark
jupyter notebook notebooks/03_tps_benchmark.ipynb
```

## License

MIT

## Citation

```bibtex
@software{aitllm2026,
  title={aitllm: Layer-Streaming LLM Inference for Low-VRAM GPUs},
  author={seanjoung},
  year={2026},
  url={https://github.com/seanjoung/-aitllm}
}
```