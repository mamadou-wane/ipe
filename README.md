# Qwen3 inference engine

`ipe` is a compact Python and PyTorch inference engine for [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B), built for logit-by-logit verification against Hugging Face Transformers and kernel-level GPU timing. vLLM and Transformers' static-cache generation serve as its performance baselines.

## Status

The engine's CPU correctness path is verified across all modules below. On NVIDIA GPUs, the 1,024-token `bf16` test suite passes on an RTX PRO 6000 Blackwell Server Edition (three sessions) and an H200 (one session). A preliminary W1 baseline has been measured on the RTX PRO 6000 against vLLM and Transformers ([docs/w1-measurements.md](docs/w1-measurements.md)), while a required harness cross-check against `vllm bench latency` remains unresolved. Workloads W2 and W3, the 4,224-token GPU suite, and kernel-level attribution of the unprofiled step time are still pending.

- [src/ipe/config.py](src/ipe/config.py): admits only the supported dense Qwen3 architecture.
- [src/ipe/checkpoint.py](src/ipe/checkpoint.py), [checkpoints/](checkpoints/): pinned checkpoint revisions, SHA-256 verification, and header-only tensor inventory inspection.
- [src/ipe/weights.py](src/ipe/weights.py): maps every checkpoint tensor to engine weights at `bf16`, `fp32`, or `fp64`, following Transformers' LM-head tying rule.
- [src/ipe/ops.py](src/ipe/ops.py), [src/ipe/forward.py](src/ipe/forward.py): decoder operators and the uncached forward pass, whose `fp64` mode serves as the correctness oracle.
- [src/ipe/engine.py](src/ipe/engine.py): prefill and decode over a preallocated KV cache, greedy and temperature sampling, and generation for static batches of equal-length prompts.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11 to 3.14.

```bash
uv sync --locked                    # PyTorch 2.13.0, Transformers 5.17.0 (tests only)
uv sync --locked --extra bench      # Linux: adds vLLM 0.31.0, which the lock satisfies
```

Checkpoints are pinned in [checkpoints/](checkpoints/) by byte size and SHA-256 hash. Download a revision into `weights/` and verify it (`HF_HOME` keeps the download client's cache inside the ignored `hf-cache/` directory):

```bash
HF_HOME="$PWD/hf-cache" uv run --locked hf download Qwen/Qwen3-1.7B \
    --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e --local-dir weights/qwen3-1.7b
uv run --locked python -m ipe.checkpoint verify checkpoints/qwen3-1.7b.json weights/qwen3-1.7b
uv run --locked python -m ipe.checkpoint inspect weights/qwen3-1.7b
```

Qwen3-0.6B (revision `c1899de289a04d12100db370d81485cdf75e47ca`, downloaded into `weights/qwen3-0.6b`) serves as the fast test fixture. Weight loading checks tensor structure without re-hashing files, so run `verify` once after each download.

## Usage

```python
import torch
from tokenizers import Tokenizer
from ipe.engine import Engine
from ipe.weights import load_checkpoint

tokenizer = Tokenizer.from_file("weights/qwen3-1.7b/tokenizer.json")
loaded = load_checkpoint("weights/qwen3-1.7b", dtype=torch.bfloat16, device="cuda")
engine = Engine(loaded, batch=1, capacity=1024)      # K/V and rotary tables allocated here
prompt = torch.tensor([tokenizer.encode("The capital of France is").ids])
result = engine.generate(prompt, 32)                 # greedy; temperature=, seed=, stop_ids= optional
print(tokenizer.decode(result.tokens[0].tolist()))
```

`engine.prefill(token_ids)` and `engine.decode(tokens)` execute single steps and return `fp32` logits for the final position. `ipe.forward.forward_full(loaded, token_ids, dtype=...)` runs the uncached forward pass, and `final_hidden` with `lm_logits` at `dtype=torch.float64` provides the numerical oracle. Sampling parameters are always passed explicitly, and the checkpoint's `generation_config.json` is ignored.

A decode step indexes preallocated buffers with host integers, avoids reading device values back to the host, and calls PyTorch's scaled dot-product attention without an explicit mask so the FlashAttention kernel remains eligible on CUDA. When timing decode, pass token IDs that are already on the device, since `decode` bounds-checks host IDs and copies them synchronously (`prefill` validates its IDs on either device). The KV cache holds one static batch at a time: `prefill` starts a new batch, attention never reads past the committed length, and a failed `decode` step leaves the committed prefix intact.

## Tests

```bash
make lint                                                   # ruff
make test                                                   # tests that need no PyTorch, GPU or checkpoint
uv run --locked pytest -m "not gpu and not checkpoint"      # adds the PyTorch tests on synthetic checkpoints
uv run --locked pytest -m checkpoint                        # needs both checkpoints downloaded
uv run --locked pytest -m gpu -k "not 4224"                 # needs an NVIDIA GPU and Qwen3-1.7B
```

Setting `IPE_TEST_MPS=1` enables device tests on Apple MPS. Some checkpoint tests require roughly 14 GB of RAM per process. Omitting `-k "not 4224"` includes the 4,224-token GPU check, which has not yet run on a GPU.

IPE uses two verification rules:

- **Bitwise parity:** Where operations run in the same order, outputs must match Transformers 5.17.0 bit for bit. On an Apple M4 Pro CPU, uncached forward passes match layer by layer for Qwen3-1.7B and Qwen3-0.6B at `fp32` and `bf16`, including sequences up to 4,224 tokens at `fp32`. Teacher-forced cached decoding also matches Transformers' cached implementation for Qwen3-1.7B (`bf16` over 1,024 tokens and `fp32` over 4,224 tokens across both attention backends), Qwen3-0.6B (256 tokens), and synthetic grouped-attention checkpoints.
- **Numerical accuracy:** Where execution orders legitimately differ, such as cached versus uncached decoding or SDPA versus math attention, `tests/needs_torch/accuracy.py` bounds residuals against an `fp64` oracle and reference variants. At `fp32`, the suite accepts valid floating-point reorderings while catching eight seeded defects.

Because some `bf16` RMSNorm defects fall inside the end-to-end tolerance, `tests/needs_torch/test_rms_norm.py` tests the operator directly against exact reference values, a contract that will also govern future fused kernels.

On CUDA, `tests/needs_torch/test_gpu.py` and the CUDA RMSNorm cases (13 tests with `-k "not 4224"`) verify `bf16` cached decoding against an on-GPU `fp64` oracle across both attention backends, alongside 32 decode steps with zero host-device synchronizations detected by PyTorch's sync debug mode and zero allocator growth, SDPA execution without falling back to the math kernel, reproducible sampling, and the TF32 guard.

Bitwise uncached parity with Transformers does not hold on either tested GPU. On the RTX PRO 6000, the divergence was isolated to rotary table construction: the engine computes its `cos` and `sin` tables on the host CPU while Transformers computes them on the GPU. Across the `fp32` tables, 19,014 `cos` and 13,806 `sin` entries differ by 1 ULP, and two entries still differ after rounding to `bf16` (`cos` at position 593, dimensions 11 and 75). As a result, uncached logits match bit for bit through row 592 and first diverge at row 593. An external diagnostic that injected Transformers' GPU rotary tables into the engine restored bitwise-identical uncached logits and layer outputs across all 1,024 rows for both attention backends. The H200 bitwise mismatch was recorded without root-cause diagnosis, and the engine continues to build rotary tables on the host.

## Benchmarks

[bench/run.py](bench/run.py) times fixed-work generation across the engine, Transformers (dynamic and static cache), and vLLM, running one implementation per process and writing raw per-iteration timings to the untracked `results/` directory:

| Workload | Batch | Prompt | New tokens |
| --- | --- | --- | --- |
| W1 | 1 | 512 | 256 |
| W2-4, W2-16 | 4, 16 | 512 | 256 |
| W3 | 1 | 4,096 | 128 |

```bash
uv run --locked python bench/run.py run --impl engine --model weights/qwen3-1.7b --workloads W1 --out results/run1
uv run --locked python bench/run.py summarize results/run1
```

Each process runs one cold call followed by 10 warm-up and 30 measured pairs of $\text{latency}(N)$ and $\text{latency}(1)$, timed from a synchronized start until device completion. Time per output token (TPOT) is $(\text{latency}(N) - \text{latency}(1)) / (N - 1)$ for every implementation, while per-step host latency is collected only in separate step-synchronized runs. Existing result files are never overwritten.

To cross-check the harness against vLLM's own benchmark tool, `vllm-calibration` runs vLLM inside `bench/run.py` using the sampling parameters and prompt distribution of `vllm bench latency`. The two tools agree when their pooled $\text{latency}(256)$ medians differ by less than the larger of their across-launch IQRs (`summarize` records the verdict in `vllm_bench_latency_agreement.agree` and exits 0 on disagreement). No GPU run with three launches per tool has passed this cross-check; a single two-launch exploratory run passed only because clock-state mixing widened its IQR tolerance.

Preliminary W1 baseline on one RTX PRO 6000 Blackwell Server Edition (one session, unlocked GPU clocks, medians over 3 process launches and 6 for the engine, with the `vllm bench latency` cross-check still unresolved):

| Implementation | latency(256) ms | TPOT ms |
| --- | --- | --- |
| engine | 3,334.8 | 13.02 |
| Transformers, dynamic cache | 3,985.9 | 15.57 |
| Transformers, static cache | 1,436.4 | 5.55 |
| vLLM | 831.0 | 3.22 |

An in-session A/A comparison between the engine's odd and even launches gave an observed engine A/A tolerance of 2.3% of $\text{latency}(256)$. See [docs/w1-measurements.md](docs/w1-measurements.md) for the full methodology, quartiles, unlocked and locked-clock cross-check diagnostics, static-cache experiments, and limitations. Raw timing dumps are kept outside the repository.

[bench/session.sh](bench/session.sh) defines the GPU session stages (`screen`, `probe`, `long`, and `bench`), and a failed correctness check stops a stage before any profiling or timing. `screen` and `probe` have run on GPUs, while `long` and `bench` have not (the preliminary W1 session ran the underlying `bench/run.py` and `vllm bench latency` commands in a custom order with six engine launches). When an instance lacks a CUDA 13 toolkit (`nvcc`, Nsight Compute, and Compute Sanitizer) and has `gcc` 6 to 15, `screen` installs the user-space CUDA 13.0.3 toolkit into `$HOME`. [bench/probes.py](bench/probes.py) implements environment capture, copy-bandwidth and launch-cost microbenchmarks, and decode profiling.

[bench/brev_session.sh](bench/brev_session.sh) was written to provision a Brev instance within a tracked time and budget ledger, run one stage, fetch a SHA-256-verified result archive, and delete the instance. It is incompatible with `brev-cli` 0.6.337 and is unmaintained: it failed to retrieve an archive in all six attempts (two instance-creation failures and four `--host` SSH route timeouts, one after the stage had run; the same route also timed out on AWS), passes `brev copy` a relative remote path that `brev-cli` rejects, treats some malformed `brev ls` listings as confirmed deletion, and refuses instance types with separately billed disks such as AWS `g7e.2xlarge`. The RTX PRO 6000 sessions instead used Brev's default SSH route under an external supervised script, and the H200 session ran `bench/session.sh probe` in a container on another provider.

## Scope

IPE targets the pinned Qwen3-1.7B checkpoint in `bf16` on a single NVIDIA GPU (compute capability $\ge 8.0$, $\ge 24\text{ GB}$ VRAM).

The engine implements transformer forward execution, preallocated KV caching, static-batch prefill and decode, and greedy and temperature sampling. PyTorch supplies the matrix multiplication and attention kernels, `safetensors` handles checkpoint reads, and Transformers serves as the reference implementation.

Model serving, paged KV allocation, continuous batching, multi-GPU execution, quantization, and speculative decoding are outside scope.

## License

MIT. See [LICENSE](LICENSE). Provenance and licensing for test fixtures are documented in [tests/fixtures/README.md](tests/fixtures/README.md).