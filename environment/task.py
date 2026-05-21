import random
from enum import Enum
from typing import List, Set

from environment.model_catalog import CATALOG

class TaskStatus(Enum):
    WAITING = 1   # 依赖未满足
    READY = 2     # 依赖已满足，等待调度
    QUEUED = 3    # 已加入服务器队列，等待资源
    RUNNING = 4   # 正在执行
    COMPLETED = 5 # 已完成


class TaskKind(Enum):
    """M2: 任务类型，区分通用 DAG 节点与 LLM 推理两阶段。"""
    GENERIC = 1   # 通用 DAG 任务（M1 及之前的所有任务，默认值）
    PREFILL = 2   # LLM 推理 prefill 阶段：处理 prompt，产出 KV cache
    DECODE  = 3   # LLM 推理 decode 阶段：逐 token 生成，依赖 prefill 的 KV cache


class Task:
    def __init__(self, task_id: int,
                 compute_demand: float,  # 并发GPU算力占用 (TFLOPS)，用于资源分配检查
                 workload: float,        # 总计算工作量 (TFLOPS)，用于计算执行时间
                 input_size: float,      # 输入数据/模型内存占用 (GB)
                 output_size: float,     # 输出数据大小 (GB)，同时决定传输时间
                 dependencies: List[int],# 依赖任务ID列表
                 priority=1,
                 model_id: str = None,   # M1: AIGC 推理模型 ID；None 表示通用任务（向后兼容）
                 kind: 'TaskKind' = None,# M2: 任务类型；None 默认为 GENERIC
                 prompt_tokens: int = 0, # M2: PREFILL 的 prompt 长度
                 output_tokens: int = 0, # M2: DECODE 期望生成的 token 数
                 req_id: int = None,     # M2: 同一推理请求的 prefill/decode 共享此 ID
                 kv_cache_GB: float = 0.0 # M2 step3: KV cache 显存占用，
                                          # prefill 构建中 + decode 使用中均占用
                ):
        self.task_id = task_id
        self.compute_demand = compute_demand
        self.workload = workload
        self.input_size = input_size
        self.output_size = output_size
        self.dependencies = dependencies
        self.priority = priority
        self.model_id = model_id  # M1: 指向 model_catalog.CATALOG 中的模型
        # M2 新增字段
        self.kind = kind if kind is not None else TaskKind.GENERIC
        self.prompt_tokens = prompt_tokens
        self.output_tokens = output_tokens
        self.req_id = req_id
        self.kv_cache_GB = kv_cache_GB  # M2 step3: KV cache 占的显存（GB）

        self.status = TaskStatus.WAITING
        self.ready_time = None      # 依赖满足、进入READY状态的时刻
        self.start_time = None
        self.end_time = None
        self.assigned_server = None  # 分配的服务器
        self.transfer_delay = 0.0   # 数据传输延迟（秒），由调度器设置
        self.cold_load_delay = 0.0  # M1: 模型冷加载延迟（秒），由 server 设置
    
    def check_dependencies(self, completed_tasks: Set[int]) -> bool:
        return all(dep in completed_tasks for dep in self.dependencies)
    
    def __lt__(self, other):
        return self.task_id < other.task_id

    # ---------- 复杂DAG结构：15任务模板，并联重复 ----------
    @staticmethod
    def generate_single_dag(task_id_offset: int = 0, num_tasks: int = 15) -> List['Task']:
        """
        生成复杂DAG结构，支持任意任务数量
        通过重复15任务模板并联实现，每个模板是独立的DAG子图
        """
        tasks = []
        current_id = task_id_offset
        remaining = num_tasks
        block_size = 15
        
        while remaining > 0:
            current_block_size = min(block_size, remaining)
            block_tasks = Task._generate_single_dag_block(current_id, current_block_size)
            tasks.extend(block_tasks)
            current_id += current_block_size
            remaining -= current_block_size
        
        return tasks

    @staticmethod
    def _generate_single_dag_block(start_id: int, num_tasks: int) -> List['Task']:
        """生成单个15任务DAG块（独立子图，内部有依赖，不连接外部）"""
        dep_map = {
            0: [],
            1: [0], 2: [0], 3: [0],
            4: [1], 5: [1], 6: [2], 7: [2], 8: [3], 9: [3],
            10: [4, 6], 11: [5, 8], 12: [7, 9],
            13: [10, 11], 14: [12, 13]
        }

        tasks = []
        for local_id in range(num_tasks):
            global_id = start_id + local_id
            deps = [start_id + d for d in dep_map.get(local_id, []) if d < num_tasks]
            task = Task(
                task_id=global_id,
                compute_demand=random.uniform(3, 20),
                workload=random.uniform(100, 2000),
                input_size=random.uniform(2, 8),
                output_size=random.uniform(0.05, 0.3),
                dependencies=deps
            )
            tasks.append(task)
        return tasks

    # ---------- 链式结构：10任务一组，并联重复 ----------
    @staticmethod
    def generate_linear_dag(task_id_offset: int = 0, num_tasks: int = 10) -> List['Task']:
        """
        生成链式结构，支持任意任务数量
        每10个任务组成一个独立的链式DAG子图，子图之间无依赖
        """
        tasks = []
        current_id = task_id_offset
        remaining = num_tasks
        block_size = 10
        
        while remaining > 0:
            current_block_size = min(block_size, remaining)
            
            for i in range(current_block_size):
                global_id = current_id + i
                deps = [global_id - 1] if i > 0 else []
                task = Task(
                    task_id=global_id,
                    compute_demand=random.uniform(3, 20),
                    workload=random.uniform(100, 2000),
                    input_size=random.uniform(2, 8),
                    output_size=random.uniform(0.05, 0.3),
                    dependencies=deps
                )
                tasks.append(task)
            
            current_id += current_block_size
            remaining -= current_block_size
        
        return tasks

    # ---------- Fork-Join结构：9任务模板，并联重复 ----------
    @staticmethod
    def generate_fork_join_dag(task_id_offset: int = 0, num_tasks: int = 9) -> List['Task']:
        """
        生成Fork-Join结构，支持任意任务数量
        通过重复9任务模板并联实现，每个模板是独立的DAG子图
        """
        tasks = []
        current_id = task_id_offset
        remaining = num_tasks
        block_size = 9
        
        while remaining > 0:
            current_block_size = min(block_size, remaining)
            block_tasks = Task._generate_fork_join_block(current_id, current_block_size)
            tasks.extend(block_tasks)
            current_id += current_block_size
            remaining -= current_block_size
        
        return tasks

    @staticmethod
    def _generate_fork_join_block(start_id: int, num_tasks: int) -> List['Task']:
        """生成单个9任务Fork-Join块（独立子图）"""
        dep_map = {
            0: [],
            1: [0], 2: [0], 3: [0],
            4: [1, 2],
            5: [3, 4],
            6: [5], 7: [5],
            8: [6, 7]
        }

        tasks = []
        for local_id in range(num_tasks):
            global_id = start_id + local_id
            deps = [start_id + d for d in dep_map.get(local_id, []) if d < num_tasks]
            task = Task(
                task_id=global_id,
                compute_demand=random.uniform(3, 20),
                workload=random.uniform(100, 2000),
                input_size=random.uniform(2, 8),
                output_size=random.uniform(0.05, 0.3),
                dependencies=deps
            )
            tasks.append(task)
        return tasks

    # ================================================================
    # M2: LLM 推理请求工厂（prefill → decode 两阶段）
    # ================================================================

    @staticmethod
    def generate_inference_request(req_id: int, task_id_offset: int,
                                    model_id: str,
                                    prompt_tokens: int,
                                    output_tokens: int) -> List['Task']:
        """生成一个 LLM 推理请求 → 返回 [prefill_task, decode_task]。

        关键设计：prefill.output_size 直接编码 KV cache 字节数。这样若
        decode 被调度到与 prefill 不同的服务器，仿真器原有的
        transfer_time = output_size / bandwidth 公式就自然把 KV 迁移代价
        算进去了——序列亲和性约束无须额外代码就成立。
        """
        if model_id not in CATALOG:
            raise ValueError(f"Unknown model_id: {model_id}")
        model = CATALOG[model_id]
        if model.family != "LLM":
            raise ValueError(
                f"generate_inference_request only supports LLM models, "
                f"got family={model.family} (model_id={model_id})")
        if prompt_tokens <= 0 or output_tokens <= 0:
            raise ValueError(
                f"prompt_tokens and output_tokens must be > 0, "
                f"got prompt={prompt_tokens}, output={output_tokens}")

        # ---- 计算量（TFLOPS）----
        prefill_workload = (prompt_tokens / 1000.0) * model.prefill_tflops_per_ktoken
        decode_workload = (output_tokens / 1000.0) * model.decode_tflops_per_ktoken

        # ---- KV cache 大小（GB），作为 prefill 的 output_size ----
        kv_cache_GB = prompt_tokens * model.kv_cache_MB_per_token / 1024.0

        # ---- 资源占用 ----
        # compute_demand：prefill 计算密集（取较高值），decode 内存密集（取较低值）
        prefill_compute = min(model.weights_GB / 4.0, 15.0)
        decode_compute = max(prefill_compute / 4.0, 1.0)
        # 激活值显存（除模型权重外的工作集）
        prefill_activation = max(0.5, 0.1 * model.weights_GB)
        decode_activation = 0.5

        prefill_id = task_id_offset
        decode_id = task_id_offset + 1

        prefill = Task(
            task_id=prefill_id,
            compute_demand=prefill_compute,
            workload=prefill_workload,
            input_size=prefill_activation,
            output_size=kv_cache_GB,         # ← KV cache 即跨机迁移代价
            dependencies=[],
            model_id=model_id,
            kind=TaskKind.PREFILL,
            prompt_tokens=prompt_tokens,
            req_id=req_id,
            kv_cache_GB=kv_cache_GB,         # M2 step3: 构建期间占显存
        )

        decode = Task(
            task_id=decode_id,
            compute_demand=decode_compute,
            workload=decode_workload,
            input_size=decode_activation,
            output_size=0.001,               # 输出文本极小（KB 级）
            dependencies=[prefill_id],
            model_id=model_id,
            kind=TaskKind.DECODE,
            output_tokens=output_tokens,
            req_id=req_id,
            kv_cache_GB=kv_cache_GB,         # M2 step3: decode 期间持续使用
        )

        return [prefill, decode]

    @staticmethod
    def generate_inference_workload(num_requests: int, task_id_offset: int,
                                     rng: random.Random,
                                     model_ids: List[str] = None,
                                     prompt_range: tuple = (64, 1024),
                                     output_range: tuple = (50, 500)
                                     ) -> List['Task']:
        """批量生成 num_requests 个独立的 LLM 推理请求。

        每个请求是 (prefill → decode) 二节点子 DAG，请求之间相互独立。
        prompt/output 长度近似真实推理服务（M4 会替换为 Azure/BurstGPT trace）。
        """
        if model_ids is None:
            model_ids = [mid for mid, m in CATALOG.items() if m.family == "LLM"]
        if not model_ids:
            raise ValueError("No LLM model available in model_ids")

        all_tasks = []
        next_id = task_id_offset
        for req_id in range(num_requests):
            model_id = rng.choice(model_ids)
            prompt = rng.randint(*prompt_range)
            output = rng.randint(*output_range)
            pair = Task.generate_inference_request(
                req_id=req_id,
                task_id_offset=next_id,
                model_id=model_id,
                prompt_tokens=prompt,
                output_tokens=output,
            )
            all_tasks.extend(pair)
            next_id += len(pair)
        return all_tasks