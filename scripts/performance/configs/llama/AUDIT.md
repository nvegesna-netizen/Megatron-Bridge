# Llama3 Config Audit Log

Tracks findings and changes from systematic audits of the llama3 performance
configs. Each entry records what was found, why it was wrong or suboptimal,
what was changed, and what still needs validation.

---

## 2026-06-16 — Llama3 70B flow audit

Scope: all GPU clusters × all precisions for 70B pretrain (V1 and V2).

### Fixed: GB300 FP8_CS missing `nccl_ub=True`

**File:** `llama3_workload_base_configs.py`
**Configs:** `LLAMA3_70B_PRETRAIN_CONFIG_GB300_FP8_CS_V1/V2`

**Finding:** GB300 BF16 V1/V2 set `nccl_ub=True` (NCCL symmetric kernels for
DP all-reduce). The equivalent GB300 FP8_CS configs use the same FSDP path
but were missing `nccl_ub=True`. `_set_nccl_ub_overrides()` explicitly
supports FSDP + nccl_ub and automatically sets `fsdp_manual_registration=True`
and `average_in_collective=False` (required for NCCL symmetric kernels). DP
all-reduce is orthogonal to FP8 TP comms — no FP8-specific incompatibility.

**Fix:** Added `nccl_ub=True` to both V1 and V2 FP8_CS configs.

**Note:** B300 FP8_CS was checked — B300 BF16 also lacks `nccl_ub`, so there
is no asymmetry there and no change is needed for B300.

**Status:** Mechanically verified. No empirical validation required beyond
what already runs for BF16 — same code path, same FSDP overrides.

---

### Added (PERF_EXPERIMENT): B200 FP8_MX CUDA graphs

**File:** `llama3_workload_base_configs.py`
**Configs:** `LLAMA3_70B_PRETRAIN_CONFIG_B200_FP8_MX_V1/V2`

**Finding:** B200 BF16 V1/V2 use `cuda_graph_impl="local",
cuda_graph_scope="full_iteration"` with TP=2, PP=4, VPP=5. B200 FP8_MX V1/V2
share the same TP/PP/VPP but had no CUDA graph settings, falling back to the
recipe default. Evidence that this is safe:

- `_set_cuda_graph_overrides()` has no FP8_MX-specific guards or assertions.
- Llama3 8B FP8_MX on B200 already uses local full-iteration CUDA graphs (8B
  FP8_MX config is aliased to the FP8_CS config which has them enabled).
- B200 FP8_MX uses CP=1 (no context parallelism) — simpler than BF16's CP=2.
- The `overlap_param_gather_with_optimizer_step` suppression for FP8_MX in
  `set_post_overrides()` is independent of CUDA graph configuration.

**Fix:** Added `cuda_graph_impl="local", cuda_graph_scope="full_iteration"` to
both V1 and V2.

**Status:** Needs empirical validation — throughput improvement and no loss
divergence / NaN gradient norms at 70B FP8_MX scale on B200. Configs are
marked with `PERF_EXPERIMENT` comments. Revert or lock as production once
validated.

---

### Known gap (not fixed): VR200 70B V1 configs missing

**File:** `llama3_workload_base_configs.py`

**Finding:** `llama3_70b_pretrain_config_vr200()` defaults to
`config_variant="v1"`, but only V2 aliases exist for VR200 70B (BF16, FP8_MX,
NVFP4). Calling the wrapper without `config_variant="v2"` crashes at runtime.
Additionally, FP8_CS has no VR200 alias at any variant.

**Fix needed:** Add V1 aliases pointing to GB200 V1 configs (same as V2 aliases
do for V2), and add a FP8_CS V2 alias. Not fixed in this pass — needs
confirmation of intended VR200 precision support matrix.

---

### Known gap (not fixed): 16K/64K recipes use TP4-tuned userbuffers with TP=8

**File:** `src/megatron/bridge/recipes/llama/llama3.py`
**Configs:** `llama3_70b_16k_pretrain_config`, `llama3_70b_64k_pretrain_config`

**Finding:** Both long-context recipes set `tensor_model_parallel_size=8` but
initialize with `userbuffers_bf16_h100_h8192_tp4_mbs1_seqlen8192` (TP4-tuned).
No TP8-tuned userbuffer for h8192 exists in `comm_overlap.py`. The existing
TP8 buffers (e.g. `userbuffers_bf16_h100_h16384_tp8_cp2_mbs1_seqlen8192`) are
for h16384 (405B hidden size), not 70B's h8192.

**Fix needed:** Create new TP8-tuned userbuffer configs for h8192 and use them
in the 16K and 64K recipes. Deferred — requires SM tuning on target hardware.
