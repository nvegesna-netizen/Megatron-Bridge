# GLM-5, GLM-5.1, and GLM-5.2

[GLM-5](https://huggingface.co/zai-org/GLM-5), [GLM-5.1](https://huggingface.co/zai-org/GLM-5.1), and [GLM-5.2](https://huggingface.co/zai-org/GLM-5.2) are large sparse MoE language models with Multi-Latent Attention and Dynamic Sparse Attention. Megatron Bridge supports these checkpoints through the shared `GLM5Bridge`.

## Supported Variants

| Variant | Hugging Face ID | Notes |
|---------|-----------------|-------|
| GLM-5 | `zai-org/GLM-5` | MoE + MLA + DSA architecture |
| GLM-5.1 | `zai-org/GLM-5.1` | Same architecture and mapping shape as GLM-5 |
| GLM-5.2 | `zai-org/GLM-5.2` | Latest GLM-5.2 checkpoint; model card lists 753B parameters and 1M-token context |

## Architecture Notes

- `GlmMoeDsaForCausalLM` architecture with 78 transformer layers.
- First 3 layers are dense; remaining layers use MoE.
- 256 routed experts with top-8 routing and one shared expert per MoE layer.
- Uses MLA plus DSA indexer parameters (`index_head_dim`, `index_n_heads`, `index_topk`).
- Requires a `transformers` build with `GlmMoeDsaForCausalLM` support.
- DSA requires the `fast-hadamard-transform` CUDA extension and MCore support for the DSA experimental attention variant.
- MCore DSAttention currently requires `context_parallel_size == 1` and `apply_rope_fusion == False`.
- GLM-5.2 advertises one next-token prediction layer, but `GLM5Bridge` disables MTP until GLM-5 MTP weight mappings are added.

## Pretraining

Use `glm52_753b_pretrain_config` for GLM-5.2 base pretraining. The JET-LLM/Silicon performance path uses:

- `model_family: glm`
- `model_recipe_name: glm52_753b`
- supported initial precisions: `bf16`, `fp8_cs`, `fp8_mx` where the GPU supports the precision
- unsupported initial precision: `nvfp4`, because GLM-5.2 workload base presets and validation coverage are not yet present

## Examples

For conversion, inference, dependency notes, hardware requirements, and MCore DSA support notes, see the [GLM-5 examples README](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/models/glm/glm5/README.md).

## Related Implementation

- Bridge implementation: [`src/megatron/bridge/models/glm_moe_dsa/glm5_bridge.py`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/models/glm_moe_dsa/glm5_bridge.py)
- Examples: [`examples/models/glm/glm5`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/examples/models/glm/glm5)
