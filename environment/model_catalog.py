"""
model_catalog.py — AIGC 推理模型规格定义

M1 阶段引入：把"任务"从孤立的 workload 升级为"对某个模型的推理请求"。
模型权重常驻显存、冷加载有几秒到几十秒代价 —— 这是 AIGC 调度区别于通用 DAG 调度的
第一个关键物理特征。

字段语义：
  weights_GB           : 加载到显存后的常驻占用（GB）
  cold_load_sec        : 从存储到 GPU 的冷加载时间（秒）
  prefill_tflops_per_ktoken / decode_tflops_per_ktoken : LLM 两阶段计算密度（M2 启用）
  tflops_per_step / steps_default                       : 扩散模型每步计算量（M2 启用）
  max_batch_size       : 服务器同模型可拼 batch 上限（M3 启用）

M1 仅使用 weights_GB 与 cold_load_sec；其余字段为 M2/M3 预留，不影响当前行为。
"""

from dataclasses import dataclass, field
import random


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    family: str                            # "LLM" | "Diffusion" | "VLM"
    weights_GB: float
    cold_load_sec: float

    # ---- LLM 字段（M2 使用） ----
    kv_cache_MB_per_token: float = 0.0
    prefill_tflops_per_ktoken: float = 0.0
    decode_tflops_per_ktoken: float = 0.0
    max_batch_size: int = 1

    # ---- Diffusion 字段（M2 使用） ----
    tflops_per_step: float = 0.0
    steps_default: int = 0
    activation_GB_per_image: float = 0.0

    # ---- M4 step2: 内存带宽下限（s / token）----
    # 真实 LLM 推理受 GPU HBM 带宽约束（尤其 decode 阶段），即使算力无限大，
    # 单 token 也要这么长时间。数量级参考 vLLM / TensorRT-LLM 公开 benchmark。
    prefill_floor_sec_per_token: float = 0.0
    decode_floor_sec_per_token: float = 0.0


# ------------------------------------------------------------
# 模型目录
#
# 参数取值参考公开技术报告与基准测试，量级正确即可，不追求精确：
#   - LLaMA-7B / 13B / 70B 权重大小来自 HuggingFace 模型卡（FP16）
#   - 冷加载时间按 NVMe SSD 顺序读 (~3 GB/s) 估算
#   - KV-cache 单 token 占用按 hidden_dim × num_layers × 2 (K, V) × 2 byte 估算
#   - prefill / decode 计算量按 2 × params × tokens 估算
# ------------------------------------------------------------

CATALOG: dict[str, ModelSpec] = {
    "llama-7b": ModelSpec(
        model_id="llama-7b",
        family="LLM",
        weights_GB=14.0,
        cold_load_sec=5.0,
        kv_cache_MB_per_token=0.5,
        prefill_tflops_per_ktoken=14.0,
        decode_tflops_per_ktoken=0.5,
        max_batch_size=32,
        # ~50 tok/s decode, ~1000 tok/s prefill on A100-class（vLLM benchmark）
        decode_floor_sec_per_token=0.020,
        prefill_floor_sec_per_token=0.001,
    ),
    "llama-13b": ModelSpec(
        model_id="llama-13b",
        family="LLM",
        weights_GB=26.0,
        cold_load_sec=9.0,
        kv_cache_MB_per_token=0.8,
        prefill_tflops_per_ktoken=26.0,
        decode_tflops_per_ktoken=0.9,
        max_batch_size=16,
        decode_floor_sec_per_token=0.030,
        prefill_floor_sec_per_token=0.0015,
    ),
    "llama-70b": ModelSpec(
        model_id="llama-70b",
        family="LLM",
        # INT8 量化后约 70 GB；FP16 原始权重 140 GB 装不进当前云服务器 128GB 显存，
        # 用量化版本既贴近主流部署实践又能体现"仅云端可承载"的物理约束。
        weights_GB=70.0,
        cold_load_sec=25.0,
        kv_cache_MB_per_token=2.5,
        prefill_tflops_per_ktoken=140.0,
        decode_tflops_per_ktoken=5.0,
        max_batch_size=8,
        decode_floor_sec_per_token=0.080,
        prefill_floor_sec_per_token=0.005,
    ),
    "sdxl": ModelSpec(
        model_id="sdxl",
        family="Diffusion",
        weights_GB=12.0,
        cold_load_sec=4.0,
        tflops_per_step=2.5,
        steps_default=30,
        activation_GB_per_image=2.0,
    ),
}


# ------------------------------------------------------------
# 模型分配辅助函数
#
# 真实推理服务里模型流行度高度长尾（少数模型承载大多数请求），
# 用 Zipf 分布近似比均匀分配更接近现实。
# ------------------------------------------------------------

def pick_model_zipf(rng: random.Random, alpha: float = 1.2,
                    model_ids: list[str] | None = None) -> str:
    """按 Zipf(alpha) 概率分布从 CATALOG 抽一个 model_id。

    alpha 越大越倾向于头部模型。alpha=1.2 大致让头部模型占 ~45% 请求。
    """
    ids = model_ids if model_ids is not None else list(CATALOG.keys())
    # 概率 ∝ 1 / (rank ** alpha)
    weights = [1.0 / ((i + 1) ** alpha) for i in range(len(ids))]
    return rng.choices(ids, weights=weights, k=1)[0]


def assign_models_zipf(tasks, rng: random.Random,
                        alpha: float = 1.2,
                        model_ids: list[str] | None = None) -> None:
    """给一批已生成的 Task 原地分配 model_id（Zipf 分布）。

    分配粒度：以 task 为单位（同一 DAG 内不同节点可以是不同模型；
    M2 阶段引入 InferenceRequest 后再改为请求级分配）。
    """
    for t in tasks:
        t.model_id = pick_model_zipf(rng, alpha=alpha, model_ids=model_ids)
