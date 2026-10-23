# W1 measurements on an RTX PRO 6000 Blackwell

At workload W1 (batch 1, 512-token prompt, 256 new tokens, `bf16`), IPE's engine takes 13.02 ms per output token on a single NVIDIA RTX PRO 6000 Blackwell Server Edition. In the same session, vLLM 0.31.0 takes 3.22 ms, Transformers 5.17.0 with its static cache takes 5.55 ms, and Transformers with its dynamic cache takes 15.57 ms. These figures form a preliminary W1 baseline from a single session with unlocked GPU clocks, and the required harness cross-check against `vllm bench latency` has not passed in any run with three launches per tool (see [Harness cross-check](#harness-cross-check)).

## Sessions

Every session ran IPE at commit `28b4c7c` from a clean working tree on an AWS `g7e.2xlarge` instance rented through Brev. Each session landed on a different physical GPU (verified by distinct GPU UUIDs), with placement selected by Brev and the region of Sessions B and C recorded from EC2 instance metadata. Measurements from separate sessions are never pooled.

| Session | Date (UTC) | Region | Work | GPU clocks |
| --- | --- | --- | --- | --- |
| A | 2026-10-17/18 | not recorded | GPU suite, rotary diagnostic, engine W1 (6 launches) | unlocked |
| B | 2026-10-21 | ap-northeast-2 | GPU suite, preliminary W1 baselines for all implementations | unlocked |
| C | 2026-10-21 | eu-west-2 | GPU suite, harness cross-check diagnostics, static-cache diagnostic | unlocked, SM clock locked for one experiment |

| Component | Value |
| --- | --- |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition, 97,887 MiB, compute capability 12.0 |
| Driver and CUDA | Driver 595.91.07, PyTorch CUDA 13.0 runtime (`2.13.0+cu130`), instance CUDA 13.2 toolkit |
| Host | 8 vCPUs on Intel Xeon Platinum 8559C, 62 GiB RAM, Ubuntu 22.04.5, kernel `6.8.0-1066-aws` (virtual machine, no container) |
| Software | PyTorch 2.13.0, Transformers 5.17.0, vLLM 0.31.0, CPython 3.12.15 (all pinned in `uv.lock`) |
| Copy bandwidth | 1,455 to 1,456 GB/s device-to-device across all sessions, counting bytes read plus bytes written (`bench/probes.py microbench`) |

## Correctness gate

Each session passed the 13-test GPU correctness suite (`pytest -m gpu -k "not 4224"`) prior to timing. The suite verifies `bf16` cached decoding over 1,024 tokens against an on-GPU `fp64` oracle across both attention backends, 32 decode steps with zero host-device synchronizations detected by PyTorch's sync debug mode and zero allocator growth, SDPA execution without the math fallback, deterministic sampling, the TF32 guard, and the CUDA RMSNorm operator contract. The same 13-test suite also passed on an NVIDIA H200 (`2026-10-17`, inside a container on another provider without W1 timing). The 4,224-token GPU check has not yet run on a GPU. See the repository [README](../README.md) for the host-versus-device rotary table difference that breaks bitwise uncached parity on the RTX PRO 6000.

## Method

- **Workload:** W1 (batch 1, 512 prompt token IDs sampled with seed 0 from the non-special vocabulary, 256 generated tokens, greedy decoding, `bf16`). The engine and both Transformers paths use SDPA attention. vLLM runs with its model defaults (`VLLM_COMPILE`, full and piecewise CUDA graphs, FlashAttention 2) and explicit flags for `bf16`, prefix caching disabled, `max_model_len` 768, `max_num_seqs` 1, and seed 0. Both `vllm-calibration` and `vllm bench latency` instead sample at temperature 1.0 using prompt IDs below 10,000 to match `vllm bench latency` (`vllm bench latency` draws a fresh unseeded prompt on each launch).
- **Per-process protocol:** `bench/run.py run --impl X --model weights/qwen3-1.7b --workloads W1 --launch k --out DIR` executes 1 cold call, 10 warm-up pairs, and 30 measured pairs of $\text{latency}(256)$ and $\text{latency}(1)$, computing $\text{TPOT} = (\text{latency}(256) - \text{latency}(1)) / 255$ per pair.
- **Launch schedule:** Three interleaved rounds in the fixed order: `engine`, `vllm`, `transformers-static`, `vllm-calibration`, `vllm bench latency`, `engine`, `transformers-dynamic` (yielding 6 launches for `engine` and 3 for every other row). `vllm bench latency` ran with the flags from `bench/session.sh`: `--model weights/qwen3-1.7b --dtype bfloat16 --input-len 512 --output-len 256 --batch-size 1 --num-iters-warmup 10 --num-iters 30 --max-model-len 768 --max-num-seqs 1 --seed 0 --disable-log-stats --no-enable-prefix-caching --disable-detokenize`.
- **Environment and warm-up:** Every process ran with `CUDA_CACHE_MAXSIZE=4294967296` (a 4 GiB persistent driver PTX JIT cache) and `VLLM_NO_USAGE_STATS=1`. vLLM's first launch took 242 s before timing began (14 s model load, 23 s `torch.compile`, 115 s CUDA graph capture, and roughly 65 s building FlashInfer's sampling module), while subsequent launches started in 16 to 18 s from warm caches. Process start-up sits outside the timed loop.
- **GPU clocks:** Unlocked. Background `nvidia-smi` sampling every 500 ms recorded the memory clock steady at 12,481 MHz with zero throttling events. Under load, the SM clock stayed between 2,280 and 2,317 MHz (98% of busy samples between 2,280 and 2,310 MHz, median 2,295 MHz), except during vLLM runs, which boosted to roughly 2,420 MHz for 1.5 to 5.5 s at the end of each `vllm` launch and for 10 to 16 s during each `vllm-calibration` and `vllm bench latency` launch (see [Harness cross-check](#harness-cross-check)).
- **Aggregation:** `bench/run.py summarize` computes pooled quartiles across all measured pairs from all launches (`statistics.quantiles`, inclusive method) and reports the across-launch IQR as the IQR of per-launch $\text{latency}(256)$ medians. Quartiles and IQR for `vllm bench latency` are computed by the same rule from its `--output-json` array (`summarize` itself prints only the `vllm bench latency` median and the agreement verdict).

An external supervised script executed these commands in the exact order above; `bench/session.sh bench` uses a 3-launch engine schedule and a different launch order.

## Preliminary W1 baselines

Session B, unlocked GPU clocks, with the `vllm bench latency` cross-check still unresolved (6.70 ms difference vs. 1.78 ms tolerance). Values are medians with pooled first and third quartiles in milliseconds:

| Implementation | Launches | latency(256) | latency(1) | TPOT | Tokens/s | Across-launch IQR |
| --- | --- | --- | --- | --- | --- | --- |
| engine | 6 | 3,334.8 [3,302.7, 3,356.4] | 13.71 [13.52, 13.82] | 13.024 [12.899, 13.108] | 76.8 | 42.3 |
| transformers-dynamic | 3 | 3,985.9 [3,971.7, 4,047.7] | 16.79 [16.68, 17.07] | 15.566 [15.510, 15.806] | 64.2 | 73.6 |
| transformers-static | 3 | 1,436.4 [1,414.9, 1,440.5] | 19.95 [19.68, 20.13] | 5.555 [5.470, 5.571] | 178.2 | 15.1 |
| vllm | 3 | 831.0 [830.2, 831.4] | 10.29 [10.24, 10.33] | 3.218 [3.215, 3.220] | 308.1 | 0.16 |
| vllm-calibration | 3 | 829.9 [822.9, 831.3] | 10.22 [9.89, 10.30] | 3.214 [3.189, 3.220] | 308.5 | 1.78 |
| `vllm bench latency` | 3 | 823.2 [823.0, 831.5] | | | | 0.05 |

$\text{Tokens/s}$ is $256 / \text{median latency}(256)$, which includes prefill. Reserved CUDA allocator bytes remained constant inside the timed loop for the engine and both Transformers paths (vLLM's internal worker process allocator is outside the harness's view). The `transformers-static` row benchmarks `generate(cache_implementation="static")` as shipped in Transformers 5.17, which instantiates a fresh `StaticCache` on each call.

## Noise floor

Comparing engine launches 1, 3, and 5 against launches 2, 4, and 6 under the `agreement()` rule in `bench/run.py` yields a $\text{latency}(256)$ median of 3,336.3 ms versus 3,334.2 ms (a 2.1 ms difference against a 77.3 ms tolerance, or 2.3% of $\text{latency}(256)$) and a TPOT difference of 0.008 ms against a 0.301 ms tolerance. This 2.3% (77.3 ms) value is the observed engine A/A tolerance for Session B, rather than a universal minimum detectable effect across all implementations. Every comparator's separation from the engine far exceeds this session tolerance: `transformers-dynamic` is +651 ms (+19.5%), `transformers-static` is -1,898 ms (-56.9%), and `vllm` is -2,504 ms (-75.1%).

Across all six engine launches, per-launch medians ranged from 3,204.7 to 3,457.4 ms (7.6%). Within launches 1, 2, 3, and 6, $\text{latency}(256)$ shifted between two plateaus separated by roughly 110 to 160 ms (shifting once in launches 2 and 6, and back and forth in launches 1 and 3), while launches 4 and 5 stayed on a single level. Throughout all six launches, the SM clock remained between 2,280 and 2,317 MHz (median 2,287 MHz). The cause of both the across-launch spread and the within-launch plateau shifts remains undetermined.

## Harness cross-check

`vllm-calibration` runs vLLM inside `bench/run.py` with the sampling settings and prompt distribution of `vllm bench latency`. Under the acceptance rule, the two tools agree when their pooled $\text{latency}(256)$ medians differ by less than the larger of their across-launch IQRs. No three-launch run has met this criterion: the tools disagreed under unlocked clocks in Session B and had an inconclusive 0.13 ms disagreement under locked SM clocks in Session C. A two-launch exploratory run in Session C passed only because clock-state mixing inflated its across-launch IQR tolerance.

| Session | Clocks and launch order | Launches per tool | Harness (ms) | `vllm bench latency` (ms) | Difference | Tolerance | Result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B | unlocked, fixed 3-round order | 3 | 829.90 | 823.21 | 6.70 | 1.78 | disagree |
| C | unlocked (`bench, harness, harness, bench`) | 2 | 828.32 | 828.92 | 0.61 | 3.29 | agree, exploratory |
| C | SM clock locked (`bench, harness, harness, bench, bench, harness`) | 3 | 838.36 | 838.49 | 0.13 | 0.09 | disagree, inconclusive |

Session C used external wrapper scripts to log wall-clock timestamps around every timed call alongside a 200 ms `nvidia-smi` sampler.

- **Session B (unlocked):** Every vLLM launch across both tools exhibited two distinct $\text{latency}(256)$ plateaus near 823 ms and 831 ms. Reconstructing per-call time windows from the end of each launch's busy period showed that 243 of the 250 single-clock-state calls landed on the matching latency level: ~823 ms at the fast clock (~2,419 MHz) and ~831 ms at the slow clock (~2,302 MHz). In every launch, the upward clock transition occurred at 43 to $44\,^\circ\text{C}$ after board power reached ~332 W. The share of fast calls tracked each launch's starting temperature, which the fixed launch order set, and the harness median landed on the slow plateau (829.90 ms) while `vllm bench latency` landed on the fast plateau (823.21 ms). Whether temperature directly triggers the clock boost remains unverified.
- **Session C (unlocked, 2 launches):** The upward clock transition again occurred at 43 to $44\,^\circ\text{C}$ and arrived earlier in warmer launches (25.4 s into load from a $37\,^\circ\text{C}$ start versus 11.0 s from a $40\,^\circ\text{C}$ start). All 110 single-state calls separated cleanly across an 831.5 ms boundary, though the slow-state tool medians still differed by 0.83 ms (835.03 ms vs. 834.20 ms) and the fast state split into two sub-levels (dropping from ~828 ms to ~823 ms about 27 s into load in three of four launches without a recorded SM or memory clock change). Overall agreement passed only because `vllm bench latency` had 21 of 30 slow calls in launch 1 and 23 of 30 fast calls in launch 2, widening its across-launch IQR tolerance to 3.29 ms.
- **Session C (SM clock locked, 3 launches):** Locking the SM clock via `nvidia-smi -lgc 2300,2300` clamped the GPU to 2,295 MHz (running at 2,280 to 2,295 MHz under load, median 2,287 MHz, with every busy sample within 20 MHz of 2,295 MHz). Every launch stayed on a single plateau with all calls within 0.41 ms of their launch median. The pooled medians differed by 0.13 ms (0.015%) against a 0.092 ms IQR tolerance. Because the 0.13 ms gap is smaller than the within-tool launch-median ranges (0.18 ms and 0.14 ms), this failure is recorded as inconclusive, and the acceptance criterion remains unchanged.

Session C's locked-clock runs tested only the two vLLM paths; no locked-clock measurements exist for the engine or Transformers, and Session C's absolute latencies are not pooled with Session B.

## Transformers static cache and CUDA graphs

In Transformers 5.17, `generate(cache_implementation="static")` allocates a new `StaticCache` on every call while executing its compiled decode step through PyTorch CUDA graph trees.

An untimed diagnostic in Session B with `TORCH_LOGS="+cudagraphs,recompiles"` (3 warm-up and 3 measured pairs, 7 $\text{latency}(256)$ calls total) recorded 7 CUDA graphs, one on the first decode step of each $\text{latency}(256)$ call (after a warm-up step in the first call), and replayed that graph for the remaining decode steps of the call. It also emitted 5,355 "Re-recording" log lines (`static input data pointer changed`), which PyTorch 2.13 prints each time it checks and skips a previously cached graph whose buffer pointers differ from the newly allocated `StaticCache`.

To bound the cost of per-call cache allocation, a single-process test in Session C (3 launches of 30 pairs, unlocked clocks) compared standard `generate(cache_implementation="static")` against passing a single preallocated `StaticCache` reset outside the timer. Allocating a fresh cache per call added 5.35 to 6.80 ms to $\text{latency}(256)$ (per-launch medians of paired differences), 2.03 to 2.14 ms to $\text{latency}(1)$, and 0.013 to 0.018 ms to TPOT (roughly 0.4% of $\text{latency}(256)$ and 0.3% of TPOT, compared with Session B's 2.3% observed engine A/A tolerance), with identical output tokens across all calls. In an instrumented pass after the timed loop, fresh-cache calls recorded zero new graphs and replayed all 255 decode steps from existing graphs; the timed loops themselves were not instrumented, so whether they captured any graphs is unknown.

## Other observations

- **Cross-session engine shift:** Session A measured the engine in isolation at a median TPOT of 12.605 ms [12.536, 12.823], $\text{latency}(256)$ of 3,227.8 ms, and an observed engine A/A tolerance of 81.2 ms (2.5%). On Session B's physical GPU and VM, the engine ran 3.3% slower (13.024 ms TPOT), showing that cross-session comparisons require re-measuring the engine in each session as a local anchor.
- **Memory-bandwidth reference:** Each W1 decode step reads roughly 3.51 GB (3.44 GB of model weights including the 0.62 GB tied LM head, plus 73 MB of average KV cache). That volume corresponds to 2.42 ms at the measured 1,455 GB/s copy rate or 2.20 ms at the GPU's 1,597 GB/s peak bandwidth specification (read-only traffic can exceed bidirectional copy throughput). vLLM's 3.22 ms TPOT is $1.33\times$ the 2.42 ms copy-rate estimate, whereas the engine's 13.02 ms TPOT is $5.4\times$ that estimate.
- **Profiled decode step:** A PyTorch profiler trace in Session A recorded 1,638 GPU kernel launches totaling 5.1 ms of device execution inside a 23.8 ms profiled decode step. Because profiler instrumentation inflates host overhead, this trace does not decompose the unprofiled 13.02 ms step directly. Unlike vLLM and `transformers-static`, which replay CUDA graphs during decoding, the engine issues eager kernel launches from Python, leaving host-versus-device attribution of the unprofiled step open.

## Limitations

- Measurements cover a single GPU model, a single workload (W1), and one comparator session with three launches per comparator. Workloads W2 and W3 and the 4,224-token GPU correctness check have not run.
- Session B ran with unlocked GPU clocks, and the ~2,420 MHz clock boost occurred only during vLLM runs, mixing clock states across implementations.
- The harness cross-check against `vllm bench latency` has not passed in any three-launch run.
- `bench/run.py summarize` outputs pooled quartiles, per-launch medians, and across-launch IQRs; it omits TPOT p90, within-launch IQRs, isolated prefill throughput, and built-in A/A tolerance reporting.
- IPE relies entirely on standard PyTorch operators with no custom CUDA kernels or CUDA graph capture.
- Raw per-iteration timing logs are stored outside this repository.