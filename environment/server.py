from enum import Enum
import heapq
from environment.task import Task, TaskStatus, TaskKind
from environment.model_catalog import CATALOG, ModelSpec


class ServerType(Enum):
    CLOUD = 1
    EDGE = 2


# ================================================================
# M3: Continuous batching 参数
#
# 当服务器同时跑多个同模型同阶段的 AIGC 任务时，它们共享 GPU；
# 单任务执行时长按
#   T_batch = T_solo × (1 + (B - 1) × overhead)
# 计算。Prefill 计算密集（compute-bound），并行收益低、overhead 高；
# Decode 内存密集（memory-bound），并行几乎免费、overhead 低。
# 数量级与 vLLM/TensorRT-LLM 公开 benchmark 一致。
# ================================================================
PREFILL_BATCH_OVERHEAD = 0.30   # 每多一个并发请求，prefill 慢 30%
DECODE_BATCH_OVERHEAD  = 0.05   # 每多一个并发请求，decode 慢 5%


class Server:
    """服务器节点。M1 改造后支持 AIGC 模型驻留显存与冷加载语义。

    显存（total_memory）现在被三个租户共享：
      1) weight_vram_used  -- 已加载的模型权重（粘性，需 LRU 驱逐释放）
      2) used_memory       -- 当前 running 任务的 input/活动显存
      3) free              -- 剩余可用

    当 task.model_id == None 时，所有 AIGC 相关逻辑短路，行为与改造前完全一致。
    """

    def __init__(self,
                 server_id: int,
                 server_type: ServerType,
                 compute_capacity: float,  # 计算能力 (TFLOPS)
                 memory: float,            # 显存容量 (GB) —— M1 后语义即 VRAM
                 storage: float,           # 存储容量 (GB)
                 bandwidth: float,         # 上行带宽 (Mbps)
                 location: str = None,     # 位置 (可选，针对边缘服务器)
                 enable_batching: bool = True  # M3 step3: 消融开关
                 ):
        self.server_id = server_id
        self.type = server_type
        self.total_compute = compute_capacity
        self.total_memory = memory
        self.total_storage = storage
        self.bandwidth = bandwidth
        self.location = location

        self.used_compute = 0.0
        self.used_memory = 0.0
        self.used_storage = 0.0

        self.running_tasks = []   # 当前运行任务
        self.task_queue = []      # 任务等待队列 (优先级队列)
        self.task_history = []    # 已完成任务记录

        # ---- M1: AIGC 模型驻留状态 ----
        self.loaded_models: dict[str, float] = {}   # model_id -> last_use_time (LRU)
        self.model_refs: dict[str, int] = {}        # model_id -> 当前 running 任务计数 (>0 表示 pinned)
        self.weight_vram_used: float = 0.0          # 已加载权重显存总量

        # ---- M3 step3: 消融开关 ----
        self.enable_batching = enable_batching

    # ================================================================
    # AIGC: 模型驻留与冷加载辅助
    # ================================================================

    def _pinned_weight_vram(self) -> float:
        """当前被 running 任务锁定、不可驱逐的权重显存总量。"""
        return sum(CATALOG[mid].weights_GB
                   for mid, refs in self.model_refs.items() if refs > 0)

    def _can_fit_model(self, model: ModelSpec, extra_activation_GB: float) -> bool:
        """检查（在允许驱逐 unpinned 模型的前提下）能否同时容纳：
        - 该模型的权重（若尚未加载）
        - 一个新 running 任务的活动显存 extra_activation_GB

        显存三租户：
          weight_vram_used (含 pinned 与 unpinned 权重)
          used_memory      (running 任务活动显存，必须保留)
          剩余             = total - weight_vram_used - used_memory
        若不够，可释放 unpinned 权重 = weight_vram_used - pinned。
        """
        pinned = self._pinned_weight_vram()
        current_free = self.total_memory - self.weight_vram_used - self.used_memory
        if model.model_id in self.loaded_models:
            # 已加载，只需要为活动显存腾地方
            return extra_activation_GB <= current_free
        # 未加载：需要塞下 weights + activation；最多能腾出 unpinned 权重
        max_free_after_eviction = self.total_memory - pinned - self.used_memory
        return model.weights_GB + extra_activation_GB <= max_free_after_eviction

    def cold_load_cost(self, model_id: str) -> float:
        """供调度器查询：把该模型加载到本服务器需要多少秒。已加载则返回 0。

        注意：此函数纯查询、不改状态。
        """
        if model_id is None or model_id not in CATALOG:
            return 0.0
        if model_id in self.loaded_models:
            return 0.0
        return CATALOG[model_id].cold_load_sec

    def _evict_lru_for(self, model: ModelSpec, extra_activation_GB: float,
                       current_time: float) -> float:
        """驱逐 LRU 的 unpinned 模型，直到当前实际剩余空间能装下所需总量。

        返回总共释放的 GB 数。若已经够用直接返回 0；若无论怎么驱逐都装不下，
        调用方应在更早的 can_allocate 阶段就拦截 —— 这里只做尽力驱逐。
        """
        need_total = (model.weights_GB if model.model_id not in self.loaded_models
                      else 0.0) + extra_activation_GB
        current_free = (self.total_memory - self.weight_vram_used
                        - self.used_memory)
        if need_total <= current_free:
            return 0.0

        # 按 last_use_time 升序排，最旧的先驱逐
        evictable = [(t, mid) for mid, t in self.loaded_models.items()
                     if self.model_refs.get(mid, 0) == 0
                     and mid != model.model_id]
        evictable.sort()

        freed = 0.0
        for _, mid in evictable:
            if need_total <= current_free + freed:
                break
            freed += CATALOG[mid].weights_GB
            self.weight_vram_used -= CATALOG[mid].weights_GB
            del self.loaded_models[mid]
            self.model_refs.pop(mid, None)
        return freed

    def _ensure_model_loaded(self, model_id: str, current_time: float) -> float:
        """确保模型已加载到本服务器；若需冷加载则执行驱逐+加载，返回冷加载耗时。

        前提：调用前已通过 can_allocate 校验过 _can_fit_model。
        """
        if model_id not in CATALOG:
            return 0.0
        if model_id in self.loaded_models:
            self.loaded_models[model_id] = current_time   # LRU touch
            return 0.0

        model = CATALOG[model_id]
        self._evict_lru_for(model, extra_activation_GB=0.0,
                            current_time=current_time)
        self.loaded_models[model_id] = current_time
        self.weight_vram_used += model.weights_GB
        return model.cold_load_sec

    # ================================================================
    # 资源分配（向后兼容：model_id is None 时走原逻辑）
    # ================================================================

    @staticmethod
    def _activation_footprint(task: Task) -> float:
        """任务运行时占用的非权重显存 = 激活值 + KV cache。

        - 通用任务: kv_cache_GB=0，等价于原来的 input_size
        - PREFILL/DECODE: input_size (激活) + kv_cache_GB (KV cache)
        """
        return task.input_size + getattr(task, "kv_cache_GB", 0.0)

    # ================================================================
    # M3: Continuous batching 辅助
    # ================================================================

    def _current_batch_size(self, model_id: str, kind: TaskKind) -> int:
        """统计当前 running 中、同 model 同 kind 的任务数（不含自身）。"""
        if model_id is None or kind == TaskKind.GENERIC:
            return 0
        return sum(1 for t in self.running_tasks
                   if t.model_id == model_id and t.kind == kind)

    @staticmethod
    def _batching_overhead(kind: TaskKind) -> float:
        """每多一个并发请求带来的相对执行时间增量。"""
        if kind == TaskKind.PREFILL:
            return PREFILL_BATCH_OVERHEAD
        if kind == TaskKind.DECODE:
            return DECODE_BATCH_OVERHEAD
        return 0.0   # GENERIC 不参与 batching

    def _batch_slot_full(self, task: Task) -> bool:
        """若 task 是 AIGC 阶段任务且其模型的 batch 已满，则拒绝接纳。

        Generic 任务永远不会被 batch 满阻塞。
        M3 step3: enable_batching=False 时永远不阻塞（消融）。
        """
        if not self.enable_batching:
            return False
        if task.kind == TaskKind.GENERIC or task.model_id is None:
            return False
        if task.model_id not in CATALOG:
            return False
        spec = CATALOG[task.model_id]
        if spec.max_batch_size <= 1:
            return False   # 该模型没启用 batching
        current = self._current_batch_size(task.model_id, task.kind)
        return current >= spec.max_batch_size

    def can_allocate(self, task: Task) -> bool:
        """检查是否满足资源需求。"""
        footprint = self._activation_footprint(task)

        # 计算与存储约束保持不变
        compute_ok = (self.used_compute + task.compute_demand
                      <= self.total_compute)
        storage_ok = (self.used_storage + task.output_size
                      <= self.total_storage)
        if not (compute_ok and storage_ok):
            return False

        # M3: Batch slot 满 → admission control 拒绝接纳
        if self._batch_slot_full(task):
            return False

        # 显存：分通用任务 vs AIGC 任务两种语义
        if task.model_id is None or task.model_id not in CATALOG:
            # 原逻辑：占用直接加到 used_memory 校验
            return self.used_memory + footprint <= self.total_memory

        # AIGC 任务：考虑权重驻留 + 可驱逐空间
        return self._can_fit_model(CATALOG[task.model_id],
                                   extra_activation_GB=footprint)

    def add_task(self, task: Task, priority: float):
        """添加任务到等待队列，标记为 QUEUED 防止重复调度"""
        task.status = TaskStatus.QUEUED
        heapq.heappush(self.task_queue, (priority, task))

    def update_resource(self, task: Task, allocate: bool):
        """更新资源分配。AIGC 任务还要维护模型引用计数。"""
        footprint = self._activation_footprint(task)
        if allocate:
            self.used_compute += task.compute_demand
            self.used_memory += footprint
            self.used_storage += task.output_size
            if task.model_id is not None and task.model_id in CATALOG:
                self.model_refs[task.model_id] = (
                    self.model_refs.get(task.model_id, 0) + 1)
        else:
            self.used_compute -= task.compute_demand
            self.used_memory -= footprint
            self.used_storage -= task.output_size
            if task.model_id is not None and task.model_id in CATALOG:
                refs = self.model_refs.get(task.model_id, 0) - 1
                if refs <= 0:
                    self.model_refs.pop(task.model_id, None)
                else:
                    self.model_refs[task.model_id] = refs

    # ================================================================
    # 任务执行
    # ================================================================

    def process_tasks(self, current_time):
        """处理任务队列，尽可能多地启动任务。

        M1 改造：AIGC 任务若需冷加载，把延迟加到 start_time 上。
        """
        while self.task_queue:
            priority, task = heapq.heappop(self.task_queue)
            if task.status != TaskStatus.QUEUED:
                # 跳过已完成或已在其他服务器运行的任务
                continue

            if self.can_allocate(task):
                # AIGC 任务：触发冷加载（含 LRU 驱逐）并取得耗时
                cold = self._ensure_model_loaded(task.model_id, current_time)
                task.cold_load_delay = cold

                task.status = TaskStatus.RUNNING
                task.assigned_server = self.server_id

                # M3: 按当前同模同阶段的并发数计算 batched 执行时长
                # batch_size 包含自己：当前 running 中同 model 同 kind 的数量 + 1
                # M3 step3: enable_batching=False 时退回 solo 行为（消融）
                solo_exec = task.workload / self.total_compute
                if self.enable_batching:
                    batch_size = self._current_batch_size(
                        task.model_id, task.kind) + 1
                    overhead = self._batching_overhead(task.kind)
                    effective_exec = solo_exec * (1.0 + (batch_size - 1) * overhead)
                else:
                    batch_size = 1
                    effective_exec = solo_exec
                task.batch_size_at_admit = batch_size

                # 执行时间 = 传输延迟 + 冷加载延迟 + batched 计算时间
                task.start_time = current_time + task.transfer_delay + cold
                task.end_time = task.start_time + effective_exec

                self.running_tasks.append(task)
                self.task_history.append(task)
                self.update_resource(task, allocate=True)
            else:
                # 资源不足，放回队列，停止本轮处理
                heapq.heappush(self.task_queue, (priority, task))
                break
