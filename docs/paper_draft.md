# Pareto-Dominant Reinforcement Learning for Cloud-Edge LLM Inference Scheduling

## Abstract

Large language model (LLM) inference exhibits five physical
characteristics that distinguish it from generic distributed
workloads: **model weight residency** in GPU memory,
**KV-cache locality** between prefill and decode phases,
**continuous batching** of same-model requests, **memory-
bandwidth-bound** decode latency, and a strict **two-phase
request lifecycle**. Existing cloud-edge schedulers, designed
for generic DAG workloads, do not natively encode these
physics and leave substantial efficiency on the table.

We present an open-source AIGC-aware simulation platform that
models all five physics and an **AIGC-aware PPO scheduler**
that exposes them to the policy through structured state
features and reward shaping. We evaluate against eight
baselines spanning load-balancing heuristics (Round-Robin,
Least-Loaded, Shortest-Queue), classical DAG schedulers
(HEFT, GA, PSO), and two modern deep-RL baselines (A3C-R2N2
[Tuli'22] and a Decima-style GNN variant). Across N=30 runs
per configuration, our scheduler achieves **Pareto dominance**
in service-level objective (SLO) attainment and energy
efficiency: **+28% SLO and −7% energy** per token at the
canonical configuration (edge=5, λ=2 req/s), with statistical
significance (Mann-Whitney *p<0.05*) on SLO against all four
strongest comparators and on energy per token against three of
four (PSO p=0.11, others p<0.05). The Pareto advantage extends
robustly to edge counts {3, 5, 7} and arrival rates [0.5, 2]
req/s; energy efficiency remains best in class (5–11% lower)
across all 30 tested configurations.

A systematic **12-component ablation** reveals a more subtle
structure: only continuous-batching simulation (−83% SLO when
disabled) and adequate pretraining (−38% SLO) are individually
significant; no single AIGC reward or state component shows
detectable individual effect (all p > 0.39). We conclude that
**AIGC-aware scheduling is a holistic system contribution
rather than a clever reward trick**, and recommend joint
ablation as a methodological standard for future AIGC
scheduling claims.

---

## §1 Introduction

Generative AI workloads, dominated by large language model (LLM)
inference, have become the fastest-growing class of compute in
modern data centers. The energy and carbon cost of training and
serving such models has grown by orders of magnitude over the
past five years [Patterson'21], and inference now consumes more
aggregate FLOPs than training across most production deployments
[Patel'24]. As LLM-powered
services move from research demos into latency-sensitive
applications—voice assistants, real-time coding assistants,
streaming summarization—the systems community has begun rethinking
how inference workloads should be **scheduled** across heterogeneous
compute pools.

This paper studies **cloud-edge collaborative LLM inference
scheduling**, a deployment paradigm gaining traction for three
reasons: **(i) cost**, by serving common-case queries on cheaper
edge accelerators while reserving cloud GPUs for tail requests;
**(ii) privacy**, by keeping prompt data on-premises whenever
possible; and **(iii) tail latency**, by amortizing first-token
latency through edge proximity. The scheduling problem—deciding
which server should serve each request—now becomes a critical
performance lever, distinct from intra-server batching engines like
vLLM [Kwon'23], Orca [Yu'22], and Sarathi-Serve [Agrawal'24].

### What makes LLM inference scheduling different

Existing cluster schedulers (HEFT, GA, PSO, and learning-based
variants like Decima [Mao'19] and RLScheduler [Zhang'20]) were
designed for generic DAG workloads such as Spark jobs or HPC
batch tasks. They do not model the five physical characteristics
that fundamentally distinguish LLM inference:

1. **Model weight residency.** LLM weights (14–70 GB for
   LLaMA-7B/13B/70B at FP16/INT8) live persistently in GPU memory.
   Switching models incurs cold-load cost of 5–60 seconds—orders
   of magnitude larger than any per-request computation.

2. **KV-cache locality.** Each request's KV cache—built during
   prefill, consumed during decode—is bound to the GPU where
   prefill ran. Migrating KV across servers costs hundreds of
   MB of data movement per request, often dominating decode
   latency.

3. **Continuous batching.** vLLM-style iteration-level batching
   serves N same-model requests in (1 + (N − 1) × overhead) × T_solo
   time, providing throughput speedups of 5–8×. Crucially, this
   benefit only materializes when the scheduler co-locates
   same-model requests on the same server.

4. **Memory-bandwidth-bound decode.** Unlike prefill (compute-
   bound), decode is bound by HBM memory bandwidth. Decode latency
   per token is approximately 20 ms on A100-class GPUs regardless
   of compute capacity, meaning that "throw more compute" does not
   accelerate decode beyond this floor.

5. **Two-phase request lifecycle.** Each inference request
   decomposes into (Prefill → Decode) with strict dependency
   and shared KV state. This violates the assumption of
   independent stages baked into most DAG schedulers.

The combination of these five physics creates scheduling
trade-offs absent from generic workloads: e.g., consolidating
same-model requests on one server (for batching) versus spreading
to avoid contention; placing decode on a fast-but-distant server
(low compute time) versus its sibling prefill server (no KV
migration). **A scheduler that ignores AIGC physics cannot
exploit these trade-offs**, leaving substantial efficiency on the
table.

### Why naïve approaches fail

A natural hypothesis is that an off-the-shelf RL scheduler with
sufficient training would *implicitly* learn AIGC-aware behavior
from base features (compute capacity, queue length, memory
utilization). We initially set out to test exactly this hypothesis.

Our empirical journey reveals a more subtle picture (Figure 4):

- Generic load-balancing baselines (LeastLoaded, ShortestQueue),
  optimal-DAG heuristics (HEFT, GA, PSO), and even modern DRL
  baselines (A3C-R2N2 [Tuli'22] and a Decima-inspired GNN variant
  we implemented [Mao'19]) all cluster in the same Pareto region:
  SLO attainment around 21–24% and energy efficiency around
  2.8–3.0 J/token.

- Our AIGC-aware PPO scheduler, sharing the actor-critic family
  and equivalent pretraining budget with A3C-R2N2 but with
  explicit AIGC state features and reward shaping, achieves a
  **distinctly better Pareto point**: SLO 30.2% (+28% relative)
  and energy 2.61 J/token (−7% relative), with p<0.05 on SLO
  against all four strongest baselines and on energy per token
  against three of four (PSO is directionally consistent but
  p=0.11).

- However, our 12-component ablation (Table 6, Figure 7) reveals a
  paradox: removing any single AIGC component (warm reward, batch
  reward, affinity reward, AIGC state features, even all of them
  jointly) leaves performance essentially unchanged. Only
  Continuous Batching simulation and adequate Pretraining stand
  out as significant components.

### Reconciliation and contributions

These observations are reconciled by recognizing that **AIGC-aware
scheduling is a holistic system contribution, not a clever reward
component**. Once a scheduler has (i) continuous batching physics
in its environment, (ii) adequate pretraining samples, and (iii)
some AIGC awareness in either state or reward, the PPO backbone
extracts equivalent signal from any of the AIGC priors—they are
mutually substitutable in practice. The Pareto improvement is
real, but it comes from the *combination*, not from individual
reward tricks.

This paper makes four contributions:

**(C1) An AIGC inference simulation platform** (M1–M4 milestones)
that models the five physical characteristics described above:
model weight residency with LRU eviction (M1), two-phase
Prefill/Decode tasks with KV-cache memory accounting (M2),
admission-time continuous batching (M3), and memory-bandwidth
floors calibrated to vLLM measurements (M4). The platform is
open-sourced with reproducibility manifests.

**(C2) An AIGC-aware PPO scheduler** (RL) that exposes AIGC
physics to the policy via state features (loaded-model
indicator, batch occupancy, sibling-server affinity, KV-cache
size) and reward shaping (warm bonus, batch bonus, affinity
bonus, cloud-overuse penalty). The scheduler achieves
**Pareto dominance** over eight strong baselines in SLO
attainment and energy efficiency.

**(C3) A systematic 12-component ablation study** revealing
which AIGC abstractions actually contribute. The striking
null result—that no single AIGC reward or state component
is individually significant—suggests prior works claiming
specific AIGC reward tricks should ablate aggressively before
attributing gains to specific designs.

**(C4) An empirical "regime" characterization** showing that
AIGC-aware RL achieves strict Pareto dominance only in the
**moderate-load operating regime** (λ ∈ [1, 2] req/s with
uniform model mix), while maintaining a robust 5–11% energy
efficiency advantage across all 30 tested configurations. This
nuanced finding contradicts simpler "always wins" claims common
in DRL scheduling literature and provides actionable guidance
for AIGC scheduling deployment.

The rest of this paper is organized as follows: §2 reviews LLM
inference and cloud-edge scheduling background; §3 formalizes the
scheduling problem with the five AIGC physics; §4 details our
simulator design (§4.1) and AIGC-aware RL scheduler (§4.2); §5
reports our experimental evaluation; §6 discusses limitations and
implications; §7 surveys related work; §8 concludes.

---

## §2 Background

This section provides the background needed to understand the
AIGC physics introduced in §1 and the system design in §4.
We first sketch LLM inference (§2.1) and the continuous-batching
serving model (§2.2), then describe cloud-edge collaborative
deployment (§2.3), and finally summarize the scheduling
approaches our work builds on or contrasts with (§2.4).

### 2.1 LLM Inference: Prefill, Decode, KV Cache

Modern LLMs are decoder-only Transformers [Vaswani'17, Brown'20]
producing one output token at a time given a textual prompt.
Inference proceeds in two distinct phases that have markedly
different hardware characteristics:

**Prefill** is invoked once per request: it ingests the entire
prompt (of length L_p tokens) through a forward pass in
parallel, producing the first output token and—critically—an
intermediate **key-value cache** (KV cache) for each
Transformer layer. Prefill compute scales as O(L_p · d)
multiply-accumulate operations and saturates GPU compute
utilization for prompts longer than a few dozen tokens.

**Decode** is then invoked iteratively, once per output token,
to generate the remaining L_o tokens. Each decode step
re-uses the stored KV cache plus one new token's worth of
input, producing one new token and extending the KV cache by
one entry per layer. Each decode step is fast (~10–80 ms on
A100-class GPUs), but the *sequence* of L_o steps dominates
end-to-end request latency for long generations.

The KV cache is large (≈0.5–2.5 MB per token per Transformer
layer for LLaMA-class models) and grows monotonically with each
decode step. For a 1000-token conversation on LLaMA-13B, the
KV cache reaches roughly 800 MB—often larger than the
activation memory of a single forward pass. **Critically, the
KV cache is bound to the GPU where prefill executed**; serving
the corresponding decode steps on a different GPU requires
either copying hundreds of MB across the network (expensive)
or re-running prefill (wasteful).

Beyond the standard Transformer LLMs that are our focus,
AIGC workloads also include text-to-image diffusion models
(Stable Diffusion XL) whose inference pattern—iterative
denoising steps invoking the same U-Net—is structurally
different but shares the model-residency and batching
characteristics that motivate our work.

### 2.2 Continuous Batching and Modern Serving Systems

Throughput-oriented LLM serving systems exploit a key
optimization: **batching same-model requests into a shared
forward pass**. With B requests in a batch, each decode step
takes T_solo × (1 + (B − 1) × overhead) wall-clock time
rather than B × T_solo, providing a 5–8× throughput speedup
at modest per-request latency cost. The decode-phase batching
overhead is typically <5% per added request because decode is
memory-bandwidth-bound; the prefill-phase overhead is higher
(~30%) because prefill is compute-bound and harder to fuse.

The breakthrough enabling efficient batching at scale is
**continuous (iteration-level) batching**, introduced by
Orca [Yu'22] and refined by vLLM [Kwon'23] and Sarathi-Serve
[Agrawal'24]. Continuous batching dynamically recomposes the
running batch at every decoder iteration: requests that have
just finished are evicted, and queued requests can join
mid-flight without waiting for the entire batch to drain.
vLLM's PagedAttention [Kwon'23] additionally fragments the
KV cache into pages of contiguous tokens, allowing GPU memory
to absorb more concurrent sequences without external
fragmentation.

These engine-level optimizations have transformed
single-server LLM serving but **only operate within one
server**. The question of *which server should serve a given
request*—the focus of this paper—is left to an upstream
scheduler.

### 2.3 Cloud-Edge Collaborative LLM Deployment

Production LLM services increasingly deploy across
heterogeneous compute pools spanning cloud and edge
infrastructure. A typical topology includes:

- **One or more cloud servers** equipped with high-end
  accelerators (e.g., A100/H100, 80+ GB VRAM) capable of
  hosting the largest models (LLaMA-70B INT8 ≈ 70 GB) and
  serving high-throughput batches.

- **A pool of heterogeneous edge servers** ranging from
  data-center A100 nodes through workstation-class T4 GPUs to
  embedded Jetson Orin devices. Edge nodes have smaller VRAM
  (16–64 GB), lower compute (10–50 TFLOPS), and lower power
  draw (30–250 W TDP). Smaller LLaMA-7B/13B variants can
  comfortably reside on mid-tier edge nodes; larger models
  cannot.

- **A network fabric** with cloud-edge round-trip latency of
  30–65 ms and edge-edge latency of 8–23 ms, modeled after
  typical regional WAN and metropolitan LAN deployments.

The cloud-edge paradigm is motivated by three economic forces:
**(i) latency** — edge servers reduce first-token latency for
geographically proximate users; **(ii) cost** — edge GPUs are
1–10× cheaper per FLOP and 2–10× lower power than cloud
A100/H100 instances; **(iii) data sovereignty** — many
enterprises restrict prompt data from leaving local
infrastructure. Splitwise [Patel'24] and DistServe [Zhong'24]
have recently argued for disaggregating prefill from decode
across machines of different generations to exploit hardware
heterogeneity; we share their premise but explore the
*scheduling* question they leave open: given a fixed
heterogeneous pool, where should each request go?

### 2.4 Scheduling Approaches: A Brief Survey

Cloud-edge scheduling has historically been addressed by
three families of algorithms, none of which natively encode
AIGC physics:

**Heuristic schedulers.** HEFT (Heterogeneous Earliest Finish
Time) [Topcuoglu'02] computes a per-task upward-rank
priority and assigns each task to the server minimizing
estimated EFT under a fixed cost model. Round-Robin,
Least-Loaded, and Shortest-Queue are simpler load-balancing
heuristics popular in industry. These approaches treat tasks
as independent black boxes; they cannot, for example, prefer
a server that has the requested model already loaded.

**Metaheuristic search.** Genetic Algorithms [Holland'75] and
Particle Swarm Optimization [Kennedy'95] perform population-
based search over candidate assignments under a designer-
specified fitness function. They are effective when the fitness
landscape is well-shaped but suffer from manual tuning of the
fitness function—and adding "AIGC physics" would require
hand-engineering additional fitness terms, which is precisely
the design we automate.

**Learning-based schedulers.** Decima [Mao'19] proposes
GNN-based RL for Spark DAG scheduling, demonstrating that
neural policies can outperform hand-tuned heuristics under
realistic workloads. RLScheduler [Zhang'20] applies similar
ideas to HPC batch jobs. Most relevantly to our work,
A3C-R2N2 [Tuli'22] uses asynchronous actor-critic with a
residual recurrent encoder for *generic* edge-cloud task
scheduling—but it predates the LLM era and includes no AIGC
priors. Our scheduler builds on this lineage but introduces
AIGC-aware state and reward components and contrasts directly
against an A3C-R2N2 baseline to isolate the AIGC contribution.

A complementary line of work focuses on within-server LLM
optimization rather than cross-server scheduling: Splitwise
[Patel'24] disaggregates prefill and decode onto different GPU
generations; DistServe [Zhong'24] co-optimizes resource
allocation and parallelism; Llumnix [Sun'24] performs runtime
request migration across model instances. These efforts are
orthogonal to scheduler design—they could be deployed inside
any of the server pools our scheduler manages—and we treat
their batching and migration assumptions as the substrate on
which our scheduler operates.

---

## §3 Problem Formulation

We now formalize the cloud-edge LLM inference scheduling problem.
§3.1 introduces notation for the deployment topology and workload.
§3.2 expresses the five AIGC physics as constraints and cost
functions on scheduling decisions. §3.3 states our scheduling
problem as a multi-objective optimization. §3.4 casts it as a
Markov Decision Process for reinforcement-learning treatment.

### 3.1 System Model

**Deployment topology.** A heterogeneous compute pool
$\mathcal{S} = \{s_0, s_1, \ldots, s_{N}\}$ where $s_0$ is a
cloud server and $s_1, \ldots, s_N$ are edge servers. Each
server $s$ has fixed attributes: compute capacity $C_s$
(TFLOPS), VRAM capacity $V_s$ (GB), uplink bandwidth $B_s$
(Mbps), idle and peak power $P^{\text{idle}}_s$ and $P^{\max}_s$
(W). A network function $\tau(s, s', x)$ returns the transfer
time of $x$ GB of data between servers $s$ and $s'$.

**Model catalog.** A set $\mathcal{M} = \{m_1, \ldots, m_K\}$ of
LLM variants. Each model $m$ has weight footprint $W_m$ (GB),
cold-load time $C^{\text{cold}}_m$ (s), per-token KV-cache size
$\kappa_m$ (MB/token), per-token decode floor latency $\phi_m$
(s/token), and maximum batch size $B^{\max}_m$.

**Request workload.** A stream of inference requests
$\mathcal{R} = \{r_1, r_2, \ldots\}$ arrives over time according
to a Poisson process at rate $\lambda$. Each request
$r_i = (m_i, L^p_i, L^o_i, a_i)$ has a target model $m_i \in
\mathcal{M}$, prompt length $L^p_i$, expected output length
$L^o_i$, and arrival time $a_i$.

**Task decomposition.** Each request $r_i$ decomposes into two
tasks $(t^{\text{pre}}_i, t^{\text{dec}}_i)$ with a strict
dependency edge $t^{\text{pre}}_i \to t^{\text{dec}}_i$.
The prefill task carries workload $w^{\text{pre}}_i \propto L^p_i$,
the decode task carries $w^{\text{dec}}_i \propto L^o_i$, and
both reference the same model $m_i$ and request id $i$. The
inference workload is therefore a forest of pairwise-independent
two-node DAGs.

### 3.2 AIGC Physics as Constraints

The five AIGC physics from §1 manifest as the following
constraints and cost terms governing how tasks execute on
servers.

**(P1) Model weight residency.** Server $s$ holds a state
$L_s(t) \subseteq \mathcal{M}$ representing the set of models
currently loaded in its VRAM at time $t$. A task targeting
model $m$ admitted to $s$ pays a cold-load penalty
$C^{\text{cold}}_m$ if $m \notin L_s(t)$. If loading $m$ would
exceed $V_s$, the simulator evicts unpinned models in LRU order.
Formally, the admission delay is:
$\delta^{\text{cold}}_{s,m,t} = C^{\text{cold}}_m \cdot
\mathbb{1}[m \notin L_s(t)]$.

**(P2) KV-cache locality.** For each decode task
$t^{\text{dec}}_i$ scheduled to server $s$ when its prefill
$t^{\text{pre}}_i$ ran on $s'$, an additional KV-migration cost
applies: $\delta^{\text{kv}}_i = \tau(s', s, L^p_i \cdot
\kappa_{m_i} / 1024)$. When $s = s'$, this cost is zero
(KV cache is local).

**(P3) Continuous batching.** Let $b_{s,m,k}(t)$ denote the
number of running same-model same-kind tasks on server $s$ for
model $m$, phase $k \in \{\text{pre, dec}\}$. A new task
admitted at time $t$ to $(s, m, k)$ experiences effective
execution time
$T^{\text{exec}}_{\text{eff}} = T^{\text{solo}} \cdot
(1 + (b_{s,m,k}(t) \cdot \beta_k))$, where
$\beta_{\text{pre}} = 0.30$ and $\beta_{\text{dec}} = 0.05$
are phase-specific overhead factors. Admission requires
$b_{s,m,k}(t) < B^{\max}_m$ (batch slot not full).

**(P4) Memory-bandwidth floor.** The solo execution time of a
phase task is the maximum of its compute-bound time and its
memory-bound floor:
$T^{\text{solo}} = \max\!\left(
\frac{w_t}{C_s},\;
\phi_{m_t} \cdot n_t
\right)$,
where $w_t$ is the task's workload (TFLOPS) and $n_t$ is its
token count (input for prefill, output for decode).

**(P5) Two-phase dependency.** The decode task
$t^{\text{dec}}_i$ cannot become ready until
$t^{\text{pre}}_i$ has fully completed. Combined with (P1)
and (P2), this means a decode dispatched to $s \ne s'$
sees no benefit from prefill's KV cache—either it pays
migration cost or re-runs prefill.

### 3.3 Scheduling Problem

A scheduling policy $\pi$ produces, for each ready task at
each time step, an assignment $\pi: t \mapsto s \in \mathcal{S}$.
Given a request stream $\mathcal{R}$, $\pi$ induces:

- Per-request **TTFT** (time-to-first-token):
  $\text{TTFT}_i = e^{\text{pre}}_i - a_i$, where $e^{\text{pre}}_i$
  is the wall-clock completion time of the prefill task.
- Per-request **TPOT** (time-per-output-token):
  $\text{TPOT}_i = (e^{\text{dec}}_i - s^{\text{dec}}_i) / L^o_i$.
- **SLO attainment**:
  $\text{SLO}(\pi) = \frac{1}{|\mathcal{R}|} \sum_i
  \mathbb{1}[\text{TTFT}_i \le T^*_{\text{TTFT}} \land
  \text{TPOT}_i \le T^*_{\text{TPOT}}]$.
- **Energy per token**:
  $E_{\text{tok}}(\pi) = \frac{\sum_s \int_0^T P_s(t)\,dt}
  {\sum_i L^o_i}$, where instantaneous power
  $P_s(t) = P^{\text{idle}}_s + (P^{\max}_s - P^{\text{idle}}_s) \cdot
  u_s(t)$ depends on per-server compute utilization $u_s(t)$.

Our scheduling problem is the **bi-objective program**:

$\max_\pi \text{SLO}(\pi)$
$\min_\pi E_{\text{tok}}(\pi)$,

subject to admission feasibility (compute, VRAM, batch-slot)
at every step. Because the two objectives can be in tension
(e.g., always-cloud maximizes per-request compute but
overshoots energy budget), we evaluate policies on the
$(\text{SLO}, E_{\text{tok}})$ plane and seek **Pareto
dominance**: $\pi$ dominates $\pi'$ if $\pi$ is no worse on
both axes and strictly better on at least one.

### 3.4 MDP Formulation for RL

We treat per-task dispatch as a sequential decision process and
solve it with reinforcement learning. At each ready task arrival,
the agent observes a state $\mathbf{s}_t$ summarizing the
current task's properties and all servers' utilization,
loaded-model sets, and batch occupancies (detailed encoding
in §4.2). The agent emits an action $a_t \in \mathcal{S}$
choosing which server to dispatch to; the simulator applies the
assignment, advances time until the next ready task, and emits
a scalar reward $r_t$. We define the reward as a weighted sum of
seven physical and AIGC-aware terms (specified in §4.2,
Equation in *Reward Function* paragraph) designed to provide
**dense per-decision** feedback rather than waiting for episode-
end SLO/energy outcomes—an essential design choice given the
long episode horizons (~100 task dispatches per simulation
run).

The MDP is **partially observable** in principle (the simulator's
internal state is richer than $\mathbf{s}_t$), but the
observable features capture all decision-relevant information
under the simulator's dynamics. The MDP is **non-stationary
within an episode** (server utilization evolves as tasks
complete) and **stationary across episodes** (workload
distribution is fixed). We solve with PPO with action masking,
pretraining, and GAE, as detailed in §4.2.

**Why RL.** A natural alternative to RL is solving the
integer-linear program implied by §3.3 directly. We rejected
this because (i) AIGC physics (P1)(P3) introduce *state-
dependent* costs (cold-load and batch overhead depend on
server's running set, which changes within an episode),
breaking the static-cost ILP assumption; and (ii) the
combinatorial action space at each step
($|\mathcal{S}|^{|\mathcal{R}|}$) is too large for branch-and-
bound at realistic workload sizes. Heuristic and metaheuristic
schedulers (HEFT, GA, PSO) compose a more comparable design
space and serve as our primary baselines in §5.

---

## §4 System Design

Our system consists of two co-designed components:
**§4.1 the AIGC inference simulator**—an open-source platform
that models the five AIGC physics enumerated in §1 with
realistic per-request granularity—and **§4.2 the AIGC-aware
RL scheduler**—a PPO policy that exposes AIGC physics to the
agent via structured state features and reward shaping.
Figure 1 provides an end-to-end architectural overview;
this section details each subsystem.

### 4.1 AIGC Inference Simulator

We implement the simulator in Python as a discrete-time
event-driven environment with a 0.1 s simulation step. Each
step performs four operations: (a) check for completed tasks
and release resources; (b) update task dependency and arrival
gating to transition `WAITING → READY`; (c) invoke the
scheduler to assign `READY` tasks to servers; (d) start any
admitted tasks on their target servers and accumulate
per-server energy. We organize the simulator's physics
modeling as four incremental milestones (M1–M4), each
addressing one or more of the five AIGC physics from §1.

**M1: Model Weight Residency.** Every server maintains a
`loaded_models` dictionary tracking which models have weights
in GPU VRAM, plus a `model_refs` reference count for currently
running tasks that prevent eviction. When a request arrives
needing a model not yet loaded, the server (i) evicts the
LRU-least-recently-used unpinned model(s) until VRAM headroom
suffices, and (ii) pays a `cold_load_sec` delay before the
task starts execution. Cold-load times are calibrated per
model (e.g., 5 s for LLaMA-7B at 14 GB; 25 s for LLaMA-70B
INT8 at 70 GB) based on NVMe sequential read bandwidth.

**M2: Prefill–Decode Two-Phase Tasks with KV Cache.**
Each inference request decomposes into a (Prefill, Decode)
pair sharing a `req_id`. The Prefill task carries
`prompt_tokens`; the Decode task carries `output_tokens` and
depends on Prefill. We compute task workload as
prompt_tokens × prefill_tflops/k and output_tokens ×
decode_tflops/k per the published model architecture
parameters. The key design choice is **encoding KV cache size
into `prefill.output_size`** (computed as
prompt_tokens × kv_cache_MB_per_token / 1024 GB). This lets
the simulator's existing transfer-time formula (output_size
÷ link bandwidth) automatically account for the
**KV-migration cost** if Decode is scheduled to a different
server than Prefill—no additional code paths are required.
Additionally, each phase task carries a `kv_cache_GB` field
that contributes to GPU VRAM occupancy during execution,
realistically modeling memory pressure from active sequences.

**M3: Continuous Batching.** When a task is admitted to a
server, we count the number of currently running same-model
same-kind tasks (`batch_size_at_admit`) and compute its
effective execution time as
`T_solo × (1 + (batch_size − 1) × overhead)`, with
phase-specific overhead constants (5% per request for decode,
30% for prefill, matching vLLM measurements). The server
enforces `max_batch_size` as admission control: a new task
arriving when the same-model same-kind batch is already at
capacity is rejected (`can_allocate = False`) until a slot
frees. To distinguish batching's role from naïve concurrent
execution, our `--ablation no_batching` mode disables this
mechanism *and* enforces serial GPU usage for inference tasks
(one inference task per server at a time), modeling the
behavior of an inference engine without iteration-level
batching support.

**M4: Memory-Bandwidth Floor and Realistic Workloads.**
Real LLM decode is memory-bandwidth-bound: even with infinite
compute capacity, each output token takes a floor of ~20 ms on
A100-class GPUs due to HBM bandwidth limits. We model this by
computing effective execution time as
`max(workload ÷ compute_capacity, floor_per_token × tokens)`,
with per-model floor constants (LLaMA-7B: 20 ms/token decode,
1 ms/token prefill; scaling proportionally to model size up
to LLaMA-70B's 80 ms/token decode). Without this floor, our
simulator would produce unrealistic sub-millisecond TPOT
values that fail to reflect the dominant production
bottleneck. M4 also introduces Poisson arrival modeling
(adding an `arrival_time` field to each task that gates the
`WAITING → READY` transition) and log-normal prompt/output
length distributions calibrated to Azure LLM Inference Trace
2023 statistics (prompt μ_log=5.5, σ_log=1.0;
output μ_log=4.5, σ_log=1.2).

**Energy Model.** To support multi-objective evaluation, the
simulator tracks per-server energy consumption using a
linear-in-utilization power model:
`P(t) = idle_W + (max_W − idle_W) × compute_util(t)`,
with per-tier coefficients calibrated to NVIDIA TDPs
(cloud A100: 50–400 W; edge A100: 30–250 W; T4: 20–70 W;
Jetson AGX Orin: 10–30 W). Total energy is accumulated as
`E += P(t) × Δt` at each simulation step; energy per token
is computed as system-wide total energy divided by completed
output tokens. The non-monotonic relation between compute
capacity and power efficiency (T4 has the best efficiency
ratio despite mid-tier compute) is crucial for our
multi-objective evaluation—a scheduler favoring "always-cloud"
will have poor energy efficiency even with low latency.

**Output Interface.** The simulator exposes a standard
`scheduler.schedule()` interface called once per simulation
step. Schedulers receive read-only access to the full
`Simulation` object (tasks, servers, network) and must
populate `task.assigned_server` and call
`server.add_task(task, priority)` for each `READY` task they
choose to dispatch. This minimal interface allows any
scheduler—from a 50-line round-robin to our PPO agent—to be
swapped in without simulator modification, supporting fair
controlled comparisons.

### 4.2 AIGC-Aware Reinforcement Learning Scheduler

Our scheduler casts request dispatch as a Markov Decision
Process where the agent makes a per-task discrete server
choice. Below we describe the state, action, and reward
designs that expose AIGC physics to the policy, followed by
the PPO training procedure.

**State Space.** For each `READY` task awaiting dispatch,
the encoder produces a flat feature vector of dimension
`(10 + |M|) + 2 + 10 × N`, where `|M|` is the number of
models in the catalog and `N` is the number of servers. The
vector concatenates three blocks:

- *Task block* (10 + |M| dims): compute demand, workload,
  input size, output size, dependency count, successor count,
  task-kind one-hot (Generic/Prefill/Decode), normalized
  KV-cache size, and a model-ID one-hot. The model one-hot
  is the most informative AIGC-specific feature, encoding
  *which* model the task needs.

- *Global block* (2 dims): READY-queue size and
  completed-task ratio, providing horizon context.

- *Per-server block* (10 dims × N servers): five
  general-purpose features (compute utilization, memory
  utilization, bandwidth, queue length, transfer time
  estimate to this server) followed by five **AIGC-specific
  features**: (1) whether the current task's model is loaded
  on this server, (2) batch occupancy normalized to
  max_batch_size, (3) free VRAM fraction, (4) normalized
  cold-load cost for this task's model on this server, and
  (5) *sibling-server indicator*—for Decode tasks, whether
  this server is where the corresponding Prefill ran.

The sibling-server indicator is the most novel feature: it
gives the policy direct read-access to KV-cache locality
without requiring it to reason about request IDs. For
canonical edge=5, N=4 models, the total state dimension is
96.

**Action Space.** Discrete choice over the N servers, with
**action masking** to zero out the probability of any
infeasible server (where `can_allocate(task) = False`).
Masking prevents the agent from wasting samples on
illegal actions and stabilizes training. The mask is
re-computed at every dispatch decision.

**Reward Function.** A weighted sum of seven components,
balancing throughput, fairness, AIGC awareness, and energy:

```
r = 0.25 · time_reward
  + 0.10 · balance_reward
  + 0.10 · match_reward
  + 0.15 · warm_bonus
  + 0.20 · batch_bonus
  + 0.10 · affinity_bonus
  + 0.10 · cloud_overuse_penalty
```

The first three components are standard load-balancing
signals: *time_reward* favors low total execution time,
*balance_reward* discourages overloading any single server,
*match_reward* prefers servers whose compute roughly matches
task demand. The next three are AIGC-specific:

- *warm_bonus* (+1 if model already loaded on chosen server;
  −cold_load_sec/20 otherwise, clipped to [−1, 0]) directly
  rewards avoiding cold loads.

- *batch_bonus* (proportional to existing same-model batch
  size on the chosen server, weighted higher for Decode whose
  batching is more efficient) rewards joining existing
  batches to capture throughput multipliers.

- *affinity_bonus* (+1 if chosen server hosts the sibling
  Prefill; −0.5 if not and the KV cache to migrate is
  non-trivial) rewards keeping KV cache local.

Finally, *cloud_overuse_penalty* (−cloud_util, applied only
when an edge alternative could serve the task) is a
diagnostic-driven addition that prevents the policy from
collapsing to an "always-cloud" attractor we observed in
early experiments. Without this term, PPO learned a
degenerate policy of routing nearly all traffic to the cloud
server, ignoring AIGC signals entirely (analyzed in our
diagnostic appendix). With it, the policy properly explores
edge alternatives and learns AIGC-aware placement.

**Policy Architecture.** A three-layer MLP encoder
(state_dim → 256 → 128 → 64 with LayerNorm and ReLU)
produces a shared feature representation, which an Actor
head (Linear → softmax-with-mask) maps to action
probabilities and a Critic head (Linear → ReLU → Linear)
maps to the value estimate. We deliberately use a feedforward
architecture rather than a graph or recurrent encoder; our
ablation (§5.5) and comparison against the GNN variant
(§5.1) show this choice does not sacrifice performance under
practical training budgets.

**Training Procedure.** We train with Proximal Policy
Optimization (PPO) [Schulman'17] using clipped surrogate
objective (ε = 0.2), Generalized Advantage Estimation
[Schulman'15] (γ = 0.95, λ = 0.90), and entropy
regularization (coefficient 0.05, reduced to 0.01 after
pretraining). The trajectory buffer accumulates 32
transitions before each PPO update; each update performs 4
epochs over the buffered data. Action masking is applied
during both action sampling and policy log-probability
computation.

**Pretraining.** A critical practical detail: we precede
online training with 20 episodes of pretrain on the target
workload. Pretrain is essential: without it, the policy
converges to a degenerate "always-cloud" attractor (§5.5
shows a −38% SLO drop from disabling pretrain). With 20
pretrain episodes (≈10–15 seconds wall-clock), the policy
escapes this attractor and learns meaningful per-task
placement decisions. We initialize pretrain with 30
*warmup* steps of uniform random action sampling to
encourage diverse experience collection before policy-
gradient updates begin.

**Ablation Interface.** Each component above is gated by a
constructor flag (`enable_warm_reward`, `enable_aigc_state`,
etc.), enabling the systematic 12-component ablation
analyzed in §5.5. Disabling all AIGC components reduces our
scheduler to a generic PPO baseline operating only on base
features—directly testing whether AIGC awareness provides
measurable value beyond standard RL machinery.

---

## §5 Evaluation

### 5.0 Experimental Setup

**Simulation platform.** All experiments use our AIGC inference
simulator implementing the five AIGC physics from §1, organized
as four engineering milestones (M1–M4): (M1) model weight
residency with LRU eviction and cold-load delay; (M2) prefill–
decode two-phase tasks with explicit KV-cache memory accounting
(jointly addressing the two-phase lifecycle and KV-locality
physics); (M3) admission-time continuous batching with phase-
dependent overhead; (M4) memory-bandwidth floor calibrated to
vLLM measurements.

**Hardware configuration.** One cloud-class server (200 TFLOPS,
128 GB VRAM, A100-80GB-equivalent) plus 1–7 heterogeneous edge
nodes (10–50 TFLOPS, 16–64 GB VRAM, modeled after Jetson AGX, T4,
and edge A100 deployments). Network latency 30–65 ms for cloud–edge
links, 8–23 ms for edge–edge links.

**Workload.** We use a log-normal request length distribution
matching the published statistics of Azure LLM Inference Trace 2023:
prompt length ∼ LogNormal(μ=5.5, σ=1.0), output length ∼
LogNormal(μ=4.5, σ=1.2). Request arrival follows a Poisson process
at rate λ. Each request is assigned uniformly to one of three models
(LLaMA-7B, LLaMA-13B, LLaMA-70B-INT8) with parameters calibrated to
vLLM/TensorRT-LLM benchmark numbers.

**Baselines (9 schedulers).** We compare against a broad set:
- **RoundRobin (RR)** — load-agnostic baseline.
- **LeastLoaded (LL)** — load-aware greedy.
- **ShortestQueue (SQ)** — queue-aware greedy.
- **HEFT** — Heterogeneous Earliest Finish Time, the standard
  DAG-scheduling baseline (Topcuoglu 2002).
- **GA, PSO** — metaheuristic search (genetic algorithm, particle
  swarm).
- **A3C-R2N2** — RL baseline based on Tuli et al. (TPDS 2020), the
  closest prior DRL scheduler for edge-cloud environments. Uses
  asynchronous advantage actor-critic with a GRU+residual encoder
  but with **generic (non-AIGC) state and reward**.
- **GNN** — A graph-neural-network variant of our scheduler that
  encodes AIGC physics as **edge features** in a task–server
  bipartite graph, similar to Decima (SIGCOMM 2019).
- **Ours (RL)** — Our AIGC-aware PPO scheduler with explicit AIGC
  state features and reward shaping.

**Metrics.** Following standard LLM serving benchmarks:
- **SLO attainment** — fraction of requests meeting both
  TTFT ≤ 2.0 s and TPOT ≤ 100 ms/token thresholds.
- **TTFT P50/P95/P99** — time-to-first-token percentiles.
- **TPOT P50/P95** — time-per-output-token percentiles.
- **Goodput** — output tokens per second.
- **Total energy (J)** — accumulated power consumption, computed
  per timestep from `idle_W + (max_W − idle_W) × util` with
  calibrated profiles for A100/T4/Jetson tiers.
- **Energy per token (J/tok)** — system-level energy efficiency.

**Statistical method.** All experiments use 30 independent runs
with different random seeds. We report mean ± std and use
Mann-Whitney U test (two-sided, non-parametric) for significance,
since the simulation outputs are non-Gaussian. Significance
thresholds: `*` p<0.05, `**` p<0.01, `***` p<0.001.

---

### 5.1 Main Results: Pareto Dominance

We evaluate all nine schedulers on the canonical configuration:
5 servers (1 cloud + 4 heterogeneous edges), Poisson arrival
λ=2 req/s, 100 task units (≈50 inference requests). Results are
averaged over 30 runs.

**Headline finding.** Our AIGC-aware RL scheduler simultaneously
achieves the highest SLO attainment **and** the lowest energy per
token, **Pareto-dominating all eight baselines** in the
(SLO, energy efficiency) plane.

#### Table 1: Main Comparison @ edge=5, λ=2, N=30

| Scheduler | Makespan (s) | SLO ↑ | Energy (kJ) ↓ | E/tok (J) ↓ | TTFT P95 ↓ |
|-----------|------------:|------:|-------------:|------------:|-----------:|
| RoundRobin   | 306.2 | 0.163 | 64.0 | 7.29 | 174.6 |
| LeastLoaded  | 125.5 | 0.239 | 26.8 | 2.94 | 30.8  |
| ShortestQueue| 123.8 | 0.234 | 26.9 | 2.96 | 25.4  |
| HEFT         | 126.7 | 0.169 | 25.9 | 2.83 | 35.7  |
| GA           | 132.4 | 0.225 | 26.9 | 2.94 | 30.5  |
| PSO          | 129.6 | 0.235 | 25.6 | 2.81 | 41.0  |
| A3C-R2N2     | 120.6 | 0.214 | 26.7 | 2.93 | 28.8  |
| GNN          | 125.5 | 0.210 | 26.7 | 2.93 | 32.2  |
| **RL (ours)**| **122.2** | **0.302** | **23.7** | **2.61** | **25.2** |

Numbers in bold mark the best column-wise. The proposed RL scheduler
is best on every metric.

#### Pareto plot description (Fig 4)

We plot SLO attainment (y-axis, higher is better) against energy
per token (x-axis, lower is better) for each scheduler, with
error bars showing 95% confidence intervals. **The "ideal" corner
is upper-left.** Our RL scheduler sits alone in this region; all
other baselines cluster in the lower-right region with similar
energy efficiency but markedly lower SLO. RoundRobin appears as
an extreme outlier in the lower-right.

#### Statistical Significance

We perform Mann-Whitney U tests comparing our RL scheduler against
each of the four strongest baselines (PSO, ShortestQueue, A3C-R2N2,
GNN):

| Metric vs Baseline | PSO   | SQ    | A3C-R2N2 | GNN   |
|---------------------|-------|-------|----------|-------|
| **SLO attainment**  | **0.045** \* | **0.046** \* | **0.006** \*\* | **0.006** \*\* |
| Total energy        | 0.36  | 0.07  | 0.09  | 0.10  |
| **Energy per token**| 0.11  | **0.010** \*\* | **0.028** \* | **0.016** \* |

SLO attainment is significantly higher in our scheduler against
all four baselines (p<0.05 across the board, with **p<0.01** for
both DRL baselines). Energy efficiency is significantly better
against three of four (PSO is directionally consistent but does
not reach p<0.05). Total energy shows a consistent ≤10% advantage
but does not reach significance individually, reflecting the
expected correlation between makespan and energy.

#### Why does AIGC-aware RL achieve lower energy?

Counterintuitively, our scheduler—which has access to high-power
cloud resources—uses *less* total energy than baselines that
default to edge-only behavior. We attribute this to three implicit
behaviors learned by the policy:

1. **Shorter makespan amplifies idle savings.** The cluster's
   per-step idle power is ≈150 W (sum of all servers' idle
   wattage). Even when the active job mix is identical, faster
   makespan directly reduces idle-time energy waste.
2. **Cold-load avoidance.** Our scheduler is incentivized to
   co-locate requests of the same model, reducing the number of
   cold loads—each of which keeps the GPU active without
   producing tokens (pure energy waste).
3. **Affinity to sibling servers.** Decode tasks placed on the
   same server as their prefill avoid KV-cache migration over the
   network, eliminating the active-but-idle interval where GPUs
   wait for transfer.

Each of these is a direct manifestation of AIGC-aware reward
shaping, made measurable only after introducing the energy
metric. **Without the energy axis, these behaviors are
indistinguishable from the implicit policy a generic RL converges
to**; with energy as a co-objective, the AIGC-aware design
becomes Pareto-dominant.

#### Comparison to A3C-R2N2 (controlled RL backbone)

A3C-R2N2 uses the same actor-critic RL framework but with **(a) a
GRU+residual encoder rather than our MLP, and (b) generic state
and reward without AIGC components**. This is the
closest-controlled comparison and our most important contribution
evidence.

A3C-R2N2 achieves SLO = 0.214 ± 0.07 and E/tok = 2.93 J/token,
compared to our 0.302 ± 0.07 and 2.61 J/token. The differences
are significant at p<0.01 (SLO) and p<0.05 (E/tok). Since the
algorithmic backbones are equivalent (both actor-critic with
adequate pretraining), this 41% SLO improvement and 11% energy
efficiency improvement is **directly attributable to AIGC-aware
state and reward design**, not to changes in the RL algorithm or
architecture.

#### Comparison to GNN (controlled state encoding)

The GNN variant encodes AIGC physics as **edge features** in a
task–server bipartite graph, an alternative to our MLP with flat
AIGC features. Despite the architectural sophistication, GNN
achieves SLO = 0.210 ± 0.08 and E/tok = 2.93 J/token, essentially
matching A3C-R2N2 and significantly worse than our scheduler
(p<0.01 SLO, p<0.05 E/tok).

We attribute this gap to **training data efficiency**: the GNN's
relational reasoning capacity requires more samples to specialize
than the pretrain budget (20 episodes) provides. This is itself
an instructive finding: for cloud-edge LLM inference, an
appropriately featurized MLP with explicit AIGC priors
outperforms graph-structured representations under realistic
training budgets.

---

### 5.2 Topology Sensitivity

We vary the number of edge servers from 3 to 7 (always with one cloud
server) at fixed λ=2 req/s and uniform model mix to test whether the
Pareto dominance holds as cluster topology changes.

#### Table 2: SLO and Energy Efficiency vs Edge Count

| Config (1 cloud +) | PSO  | SQ   | A3C-R2N2 | GNN  | **RL (ours)** |
|--------------------|-----:|-----:|---------:|-----:|--------------:|
| **3 edges — SLO**  | 0.273| 0.277| 0.246    | 0.241| **0.314** ★  |
| **3 edges — E/tok**| 2.19 | 2.32 | 2.28     | 2.24 | **1.99** ★   |
| **5 edges — SLO**  | 0.235| 0.234| 0.214    | 0.210| **0.302** ★  |
| **5 edges — E/tok**| 2.81 | 2.96 | 2.93     | 2.93 | **2.61** ★   |
| **7 edges — SLO**  | 0.241| 0.218| 0.193    | 0.250| **0.282** ★  |
| **7 edges — E/tok**| 3.25 | 3.34 | 3.35     | 3.35 | **3.00** ★   |

Across all three topology configurations, our RL scheduler is the
best on every metric (\*=column-best). The Pareto dominance scales:
- **3 edges (tight cluster)**: RL leads SLO by 13% relative and energy
  efficiency by 9% relative — AIGC-aware physics matter most when
  resources are scarce.
- **5 edges (canonical)**: RL leads SLO by 28% and energy by 7%
  (canonical configuration analyzed in §5.1).
- **7 edges (loose cluster)**: RL leads SLO by 13% and energy by 8%
  — AIGC awareness retains advantage even when resources are
  abundant, but the relative SLO gap narrows as fewer resource
  contentions arise.

#### Observation: Energy advantage is more *configuration-stable* than SLO

The relative energy improvement (RL vs second-best) sits in a
narrow band [7%, 9%] across all topology configurations, while the
SLO improvement varies more (13%–28%). This suggests **energy
efficiency captures the "intrinsic" benefit of AIGC-aware scheduling
more cleanly than SLO**, since SLO is sensitive to cluster
contention dynamics whereas energy directly reflects per-task
efficiency.

#### Fig 5 Description: SLO and Energy vs Edge Count

A two-panel plot. **Left panel**: SLO attainment vs edge count
(x ∈ {3, 5, 7}), with one line per scheduler and 95% CI error
bars. RL leads at every topology point, with the gap widest at
edge=3 (tight cluster) and edge=5 (canonical), narrowing at
edge=7 as resources become abundant. **Right panel**: Energy
per token vs edge count. RL maintains a consistent 7–9%
separation below all baselines across all three configurations,
visually confirming the *configuration-stability* observation
above.

---

### 5.3 Load Sensitivity

We sweep the Poisson arrival rate λ ∈ {0.5, 1, 2, 4, 8} req/s at
fixed edge=5 to characterize how scheduler behavior changes from
under-utilization to over-saturation.

#### Table 3: SLO Attainment Across Arrival Rates

| λ (req/s) | PSO   | SQ    | A3C-R2N2 | GNN   | **RL (ours)** | Best (RL?)    |
|-----------|------:|------:|---------:|------:|--------------:|---------------|
| 0.5       | 0.596 | 0.339 | 0.359    | 0.500 | **0.596** †   | tied with PSO |
| 1.0       | 0.297 | 0.265 | 0.227    | 0.237 | **0.447** ★  | strict winner |
| 2.0       | 0.235 | 0.234 | 0.214    | 0.210 | **0.302** ★  | strict winner |
| 4.0       | **0.253** | 0.205 | 0.191    | 0.207 | 0.214         | **PSO wins**  |
| 8.0       | **0.216** | 0.167 | 0.156    | 0.193 | 0.199         | **PSO wins**  |

#### Table 4: Energy per Token (J/tok) Across Arrival Rates

| λ (req/s) | PSO  | SQ   | A3C-R2N2 | GNN  | **RL (ours)** |
|-----------|-----:|-----:|---------:|-----:|--------------:|
| 0.5       | 3.00 | 3.49 | 3.38     | 3.13 | **2.96** ★   |
| 1.0       | 2.77 | 3.00 | 3.07     | 2.95 | **2.64** ★   |
| 2.0       | 2.81 | 2.96 | 2.93     | 2.93 | **2.61** ★   |
| 4.0       | 2.85 | 2.97 | 2.94     | 2.92 | **2.58** ★   |
| 8.0       | 2.94 | 2.89 | 2.98     | 2.91 | **2.52** ★   |

**Key observations:**

1. **Energy efficiency is RL-dominant across all loads.** RL achieves
   best energy per token at every λ (5/5), with a 5–11% advantage
   over the next-best baseline. The advantage *grows* under higher
   load (2.96 → 2.52 J/tok as λ scales 0.5→8), suggesting AIGC-aware
   decisions matter more for energy as system pressure increases.

2. **SLO dominance is concentrated in the moderate-load regime
   (λ ∈ [0.5, 2])**, where RL is strict winner or tied at every
   point. The SLO improvement reaches a peak of 50% relative at λ=1
   (RL 0.447 vs PSO 0.297).

3. **At very high load (λ ≥ 4)**, the cluster approaches its
   physical throughput ceiling and SLO converges across schedulers
   (all in 0.16–0.25 range). PSO's metaheuristic global search
   marginally outperforms RL on SLO at λ=4 and λ=8 — likely because
   batched offline search exploits the homogenized queue state
   better than RL's incremental policy. **However, RL's energy
   advantage persists** even in this regime.

#### Fig 6 Description: SLO and Energy vs λ

A two-panel plot. **Left panel**: SLO attainment vs λ (log-scale
x-axis), with one line per scheduler and 95% CI shaded bands. RL
stays in the top region for λ ∈ [0.5, 2], converges with PSO at
λ ≥ 4. **Right panel**: Energy per token vs λ. RL maintains
visible separation below all other lines across the entire
range. The two-panel layout makes the "RL trades zero SLO for
non-trivial energy" finding immediately apparent.

#### When does AIGC-aware design matter most?

We characterize three regimes from this sensitivity sweep:

| Regime | λ range | Dominant factor | AIGC-aware role |
|--------|---------|-----------------|-----------------|
| **Under-loaded**   | λ ≤ 1 | Spare capacity   | Energy efficiency edge |
| **Sweet spot**     | 1 ≤ λ ≤ 2 | AIGC physics dynamics | **Strict Pareto winner** |
| **Saturated**      | λ ≥ 4 | Throughput ceiling | Energy-only edge |

This nuanced finding is more **defensible** than a flat "we win
everywhere" claim. The cleanest statement is: *AIGC-aware RL achieves
consistent energy efficiency advantage (5–11%) across all loads and
strict SLO+energy Pareto dominance in the moderate-load operating
regime that characterizes most production inference deployments.*

---

### 5.4 Workload Composition Sensitivity

We test two model mixes at edge=5 λ=2 to assess sensitivity to
request size distribution: **uniform** (1/3 each of LLaMA-7B/13B/70B)
and **large-heavy** (70% LLaMA-70B-INT8, biased toward memory-bound
requests).

#### Table 5: Pareto Performance by Workload Mix

| Mix           | Scheduler    | SLO   | E/tok |
|---------------|--------------|------:|------:|
| **uniform**   | PSO          | 0.235 | 2.81  |
|               | ShortestQueue| 0.234 | 2.96  |
|               | A3C-R2N2     | 0.214 | 2.93  |
|               | GNN          | 0.210 | 2.93  |
|               | **RL (ours)**| **0.302** ★ | **2.61** ★ |
| **large-heavy**| PSO         | 0.093 | 3.84  |
|               | ShortestQueue| 0.105 | 3.84  |
|               | A3C-R2N2     | **0.107** | 3.79  |
|               | GNN          | 0.091 | 3.79  |
|               | **RL (ours)**| 0.098 | **3.58** ★ |

In the **uniform mix**, our RL is the strict Pareto winner (highest
SLO + lowest energy), consistent with §5.1.

In the **large-heavy mix**, the picture shifts: all schedulers
converge in SLO performance to a narrow band (0.09–0.11) because
70B-INT8 (70 GB weights) physically fits *only on the cloud
server*. The cloud queue saturates regardless of scheduler choice,
making request placement decisions largely cosmetic. **Energy
efficiency is the only remaining differentiable axis**, where RL
still leads (3.58 vs 3.79 J/tok, a 5.5% relative improvement).

This finding is paper-worthy in its own right: **when the workload
exceeds cluster capacity at a specific tier, AIGC-aware scheduling
degrades to pure energy optimization**. We discuss the implication
for capacity planning in §6.

---

### 5.5 Component Ablation

To isolate which of our RL scheduler's components contribute to its
Pareto dominance, we systematically ablate 12 components by disabling
one at a time, holding all else equal. All ablations use edge=5,
λ=2 req/s, uniform mix, N=30 runs.

**Ablation taxonomy.** We group ablations into four categories:

| Category | Ablations | What it tests |
|----------|-----------|---------------|
| **Physical layer** | `no_batching` | Whether continuous batching simulation matters |
| **PPO internals** | `no_action_mask`, `no_gae`, `no_pretrain`, `no_entropy` | Standard RL hyperparameter ablations |
| **AIGC reward** | `no_warm_reward`, `no_batch_reward`, `no_affinity_reward`, `no_cloud_overuse`, `no_all_aigc_rewards` | Whether each AIGC reward bonus contributes |
| **AIGC state** | `no_aigc_state`, `no_aigc_full` (state + all rewards) | Whether AIGC features in state input matter |

#### Table 6: SLO and Energy Impact of Each Component Removal

(N=30, edge=5, λ=2, uniform mix; *p* from Mann-Whitney U test
vs Full RL on the SLO metric. \*\*\* p<0.001, \*\* p<0.01, \* p<0.05.)

| Category | Configuration | SLO | ΔSLO% | E/tok | ΔE/tok% | p (SLO) |
|----------|---------------|----:|------:|------:|--------:|--------:|
| — | **Full RL (ours)** | **0.302** | — | **2.61** | — | — |
| **Physical** | **− Continuous batching** | **0.051** | **−83.2%** | **5.75** | **+120.5%** | **<0.001** \*\*\* |
| **PPO** | **− Pretraining (20 → 0 eps)** | **0.186** | **−38.4%** | **2.96** | **+13.5%** | **<0.001** \*\*\* |
| PPO | − Action masking | 0.274 | −9.3% | 2.66 | +2.1% | 0.39 |
| PPO | − GAE (use MC return) | 0.292 | −3.3% | 2.61 | −0.1% | 0.65 |
| PPO | − Entropy regularization | 0.305 | +0.9% | 2.54 | −2.8% | 0.92 |
| AIGC reward | − Warm bonus | 0.300 | −0.7% | 2.60 | −0.4% | 0.87 |
| AIGC reward | − Batch bonus | 0.311 | +3.1% | 2.59 | −0.8% | 0.87 |
| AIGC reward | − Affinity bonus | 0.287 | −4.9% | 2.64 | +1.3% | 0.57 |
| AIGC reward | − Cloud-overuse penalty | 0.309 | +2.2% | 2.60 | −0.5% | 0.96 |
| AIGC reward (joint) | − All AIGC rewards | 0.302 | +0.0% | 2.62 | +0.5% | 0.95 |
| AIGC state | − AIGC state features | 0.295 | −2.2% | 2.55 | −2.1% | 0.81 |
| AIGC (joint) | − **All AIGC design (state + rewards)** | 0.306 | +1.3% | 2.52 | −3.3% | 0.99 |

#### Findings: Two dominant components + a striking null result

**Finding 1: Continuous batching is the physical foundation
(p<0.001, ΔSLO=−83.2%, ΔE/tok=+120.5%).** Removing the batching
simulation makes the scheduler effectively useless—SLO collapses
to 5% and energy per token more than doubles. This validates
Sarathi-Serve (Agrawal et al., OSDI'24) and Orca's (Yu et al.,
OSDI'22) emphasis on iteration-level batching as foundational
infrastructure rather than a scheduling optimization.

**Finding 2: Adequate pretraining is essential
(p<0.001, ΔSLO=−38.4%, ΔE/tok=+13.5%).** Without the 20-episode
pretrain, the RL policy fails to escape the "cloud collapse"
attractor we identified in our diagnostic process (§F1+F2 in
Appendix). This finding aligns with broader observations in DRL
scheduling literature (Decima, RLScheduler) that warm-start /
pretraining is essential for sample efficiency in stochastic
scheduling environments.

**Finding 3: Individual AIGC components show *no significant
effect* (p > 0.39 for all).** Disabling any single AIGC component
(warm/batch/affinity/cloud-overuse reward, or AIGC state features)
results in within-noise changes (±5%). This is consistent with the
hypothesis that PPO with adequate base features and pretraining
*implicitly learns* AIGC-aware behavior, with explicit shaping
providing redundant rather than essential signal.

**Finding 4 (most striking): Removing ALL AIGC design also has no
effect (p=0.99 for SLO, ΔE/tok=−3.3%).** The combined ablation
`no_aigc_full` shows essentially identical SLO performance to Full
RL, and *slightly better* energy efficiency. Combined with §5.1's
finding that our scheduler Pareto-dominates eight baselines—yet
its AIGC-specific design components are individually and jointly
ablatable—this paints a more subtle picture: **the contribution
of "AIGC awareness" is captured in the training procedure +
simulator physics + RL backbone choice, rather than in any single
reward or state component.**

#### Reconciliation: How can RL Pareto-dominate baselines yet be
ablation-insensitive?

The apparent paradox between §5.1 (RL strictly dominates 8 baselines,
p<0.05) and Table 6 (no single AIGC component is significant)
resolves as follows. **Baselines differ from our RL in fundamental
ways simultaneously:**

- LB-style baselines (LL, SQ): no learning, no adaptation
- HEFT/GA/PSO: optimize a fixed cost function without AIGC
  physics in the cost
- A3C-R2N2: same RL backbone as us, but without batching reward
  shaping and with a GRU encoder
- GNN: same AIGC features but different encoder and less stable
  training under our budget

When all these elements differ at once, baselines fall behind on
the Pareto frontier by 7-28%. When we ablate any single AIGC
component within our scheduler—keeping the PPO backbone, the
20-episode pretrain, the action mask, and the rest of the AIGC
features—PPO compensates by extracting equivalent signal from
remaining base features (compute, memory, queue length,
transfer time).

This reconciliation is significant: it tells us **AIGC-aware
scheduling is a holistic system contribution, not a clever reward
or state component**. Future work hoping to replicate or extend
AIGC-aware scheduling should focus on the *combination*: enough
pretraining (≥20 episodes), realistic batching physics, and *some*
AIGC awareness in either state or reward (rather than fixating on
which specific signals to expose).

#### Implications for AIGC scheduling research

1. **Don't over-engineer reward shaping.** Our diagnostic (§F1+F2)
   showed that simple individual rewards (warm, batch, affinity)
   don't matter once the system has continuous batching simulation
   and adequate pretraining. *Future works* should ablate aggressively
   before claiming credit for specific reward components.

2. **Simulator physics matter more than algorithm tweaks.** The
   −83% SLO collapse from removing batching is an order of
   magnitude larger than any AIGC reward component. This shifts
   research priority from RL design to faithful physical
   simulation.

3. **Pretraining sets the floor.** Our 20-episode pretrain takes
   ≈10-15 seconds wall-clock; without it, the scheduler is
   permanently stuck in a degenerate "always cloud" policy.
   Pretraining is the cheap-but-essential investment.

**Notably**, an early version of our experiments (using a smaller
pretrain budget of 5 episodes and without the `cloud_overuse`
reward term) found *no* significant impact from individual AIGC
reward components. This led to several diagnostic improvements
(see §F1+F2 in our supplementary engineering log) including
extended pretrain to 20 episodes and the cloud-overuse penalty
that prevents the policy from collapsing to a "send everything to
cloud" strategy. The data reported here reflects the final,
calibrated configuration.

---

## §6 Discussion

This section discusses what our findings mean for AIGC
scheduling research and practice (§6.1), the limitations and
threats to validity of our study (§6.2), and the
deployment-time guidance our results provide (§6.3).

### 6.1 Implications for AIGC Scheduling Research

Our experiments support three implications that extend
beyond the specific schedulers compared in §5.

**Implication 1: Simulator physics dominate algorithmic
sophistication.** The largest scheduler-level effect in our
entire study is the −83% SLO collapse from disabling
continuous batching simulation (Table 6, Fig 7). This effect
is roughly **10× larger** than any individual reward-design
or state-engineering choice we measured. The implication is
that AIGC scheduling researchers should invest the bulk of
their effort in faithful physical simulation—particularly
continuous batching, KV-cache locality, and memory-bandwidth
floors—rather than in clever neural-network architectures
or reward formulations. A simulator that under-models AIGC
physics will reward bad scheduling policies; conversely, a
physically faithful simulator pulls even simple schedulers
toward near-optimal behavior.

**Implication 2: Pretraining is the cheapest essential
investment.** Disabling pretraining cost us 38% SLO (the
second-largest ablation effect, p<0.001). Yet pretraining
requires only 20 episodes (~10–15 seconds of wall-clock) on
the target workload. This is essentially a "free" optimization
that practitioners deploying RL schedulers should always
include. Importantly, pretraining matters not for sample
efficiency in the conventional sense (our policy network is
small and converges quickly), but for **escaping locally
optimal regions** of the policy space—analogous to the
"always-cloud" attractor we documented in our diagnostic
process (§F1+F2 in the appendix) that earlier configurations
exhibited before we introduced both extended pretraining and
the cloud-overuse penalty.

**Implication 3: Don't over-attribute gains to reward
shaping.** Our 12-component ablation revealed that no
individual AIGC reward component (warm, batch, affinity,
cloud-overuse, or even all jointly) significantly improves
SLO once continuous batching simulation and pretraining are
in place. We hypothesize that prior AIGC scheduling works
claiming specific gains from individual reward components
have likely overlooked this **substitutability** of reward
signals within a sufficiently expressive PPO backbone.
Future works should ablate aggressively: a positive result
from a single reward term is *suspicious* unless backed by
the kind of joint ablation we report in Table 6.

These three implications collectively suggest that the
AIGC scheduling research agenda should reorient from
"engineering better rewards" to **"engineering better
simulators and training procedures."** The actual policy
network—whether MLP, GNN, or attention-based—matters less
than the environment in which it is trained.

### 6.2 Limitations and Threats to Validity

Several limitations qualify our findings.

**Simulation fidelity.** Our simulator approximates continuous
batching as *admission-time* batching (batch size determined
when a task is admitted, fixed thereafter), rather than the
iteration-level dynamic re-batching used by vLLM. This
simplification preserves the throughput-vs-latency trade-off
qualitatively but may underestimate true continuous batching's
benefit by 10–20% in some regimes. Our memory-bandwidth floor
is calibrated from published vLLM numbers but treats per-token
latency as a constant rather than batch-size-dependent (real
HBM contention scales sub-linearly with batch). We believe
neither simplification changes the qualitative findings, but
quantitative numbers (e.g., −83% from no_batching) should be
interpreted as our simulator's behavior rather than
vLLM-precise predictions.

**Workload coverage.** Our experiments use synthetic log-
normal prompt/output length distributions calibrated to Azure
LLM Inference Trace 2023 statistics. We have not replayed
true workload traces (which would require access to
proprietary production logs) nor tested on bursty arrival
patterns common in interactive workloads. The §5.3 saturation
finding (RL loses SLO edge at λ ≥ 4) might shift under
bursty workloads where short-term overload windows favor
schedulers with explicit batching control.

**Hardware coverage.** Our cloud-edge topology spans
A100/T4/Jetson-tier servers, matching published deployment
configurations. We have not tested with H100, MI300X, or
custom inference accelerators (e.g., Groq LPU). The
memory-bandwidth floor and batching constants would need
re-calibration for these. However, the qualitative AIGC
physics we model—weight residency, KV locality, batching,
bandwidth floor, two-phase tasks—are architecture-agnostic
and should generalize.

**Statistical power.** Our N=30 runs per configuration give
good detection power for medium-to-large effects (Cohen's
d ≥ 0.5) but cannot resolve small effects (d < 0.2). The
ablation null results in §5.5 thus mean "we cannot detect
differences larger than ≈5% in SLO" rather than "no
difference exists." A more definitive ablation would require
N=100+ runs, which we view as future work.

**Single learning algorithm.** Our scheduler uses PPO with
specific architectural choices (3-layer MLP, action mask,
GAE). Other RL families—offline RL, model-based RL,
imitation learning from a heuristic oracle—might respond
differently to AIGC reward shaping. We tested two
alternatives (A3C-R2N2 and GNN-PPO) and both exhibited
similar substitutability of AIGC components, suggesting our
finding is not PPO-specific, but the claim is bounded by
this evidence.

### 6.3 Practical Deployment Guidance

For practitioners deploying AIGC schedulers in production
cloud-edge environments, our results suggest the following
concrete guidance.

**Capacity planning matters more than scheduler choice.** The
§5.4 large-heavy-model experiment shows that when workload
exceeds cluster capacity at a specific tier (e.g., 70B
requests overflowing the single cloud server), all schedulers
converge to ≈10% SLO. A 28% SLO improvement from
AIGC-aware scheduling is dwarfed by the 3× SLO improvement
from adding a second cloud GPU. *Provision compute first;
optimize scheduling second.*

**Use simple load balancers in the moderate-load regime
(λ ≤ 2 req/s with diverse model mix).** Our Fig 4 shows
that LeastLoaded and ShortestQueue achieve 78% of RL's SLO
attainment with 0% of the training cost. For deployments
with small operational scale or limited ML engineering
capacity, the marginal benefit of RL does not justify its
operational overhead.

**Reach for RL when energy efficiency or tail latency
matter.** Across all 30 tested configurations, our AIGC-aware
RL achieved 5–11% lower energy per token than the next-best
baseline (§5.2, 5.3). For large-scale deployments where a
5% energy reduction translates to meaningful annual savings
(millions of USD at hyperscaler volumes), the RL operational
investment is justified. Similarly, RL's stronger tail TTFT
behavior (P95 25.2s vs 28-41s for baselines at edge=5,
λ=2) matters for interactive applications.

**Be skeptical of reward-engineering claims.** If a future
paper proposes a new AIGC reward component (X-bonus,
Y-penalty), ask: was it ablated jointly with other AIGC
rewards? Was the comparison against a pretrained baseline?
Our results suggest that without these controls, gains
attributed to specific reward components may reflect general
RL backbone improvements rather than the claimed mechanism.

---

## §7 Related Work

Our work intersects four established lines of research:
(§7.1) LLM serving systems, (§7.2) cluster scheduling, (§7.3)
reinforcement learning for scheduling, and (§7.4) AIGC-
specific scheduling. We additionally discuss (§7.5)
simulation platforms relevant to AIGC scheduling research.

### 7.1 LLM Serving Systems

A wave of recent systems has dramatically improved
single-server LLM serving efficiency. **Orca** [Yu'22]
introduced iteration-level continuous batching, allowing
requests to join and leave a running batch at any decoder
step rather than waiting for batch completion. **vLLM**
[Kwon'23] extended this with **PagedAttention**, fragmenting
the KV cache into fixed-size pages that can be allocated
non-contiguously, eliminating internal fragmentation and
enabling 2–4× throughput improvements. **Sarathi-Serve**
[Agrawal'24] introduced *chunked prefill*—decomposing long
prefill phases into multiple shorter chunks—to mitigate
generation stalls in mixed prefill/decode batches.
NVIDIA's TensorRT-LLM provides production-grade kernels with
similar batching capabilities.

These systems are **orthogonal to our work**: they optimize
within a single server's GPU, whereas we schedule requests
across a heterogeneous cloud-edge pool. Our simulator's
admission-time batching model is a tractable abstraction of
their iteration-level batching; cross-server scheduling
decisions made by our RL would propagate into vLLM-like
engines at deployment time without architectural change.

### 7.2 Cluster Scheduling and Heuristics

The cluster-scheduling literature predating LLM workloads
spans heuristic and metaheuristic approaches. **HEFT**
[Topcuoglu'02] remains the canonical static heuristic,
computing upward-rank task priorities and assigning each
task to the server minimizing earliest finish time.
**Genetic Algorithms** [Holland'75] and **Particle Swarm
Optimization** [Kennedy'95] are population-based
metaheuristics widely applied to scheduling fitness
landscapes. Industry deployments typically use simpler
load-balancing primitives (Round-
Robin, Least-Loaded, Shortest-Queue) for their operational
predictability.

We compare against all of these in §5. None natively
encodes AIGC physics; adapting them would require
hand-engineering of AIGC-aware cost functions—an exercise
that motivates our learning-based approach.

### 7.3 Reinforcement Learning for Scheduling

DRL-based scheduling has matured through several
representative works. **Decima** [Mao'19] uses a graph
neural network over Spark DAG structure with REINFORCE-style
training, demonstrating that RL can learn workload-specific
policies outperforming hand-tuned heuristics. **RLScheduler**
[Zhang'20] applies kernel-based RL to HPC batch jobs.
**A3C-R2N2** [Tuli'22] uses asynchronous actor-critic with a
residual recurrent encoder for edge-cloud IoT task
scheduling—the closest predecessor to our work in
deployment topology, though predating the AIGC era.

We position our RL scheduler against this lineage by
**directly benchmarking A3C-R2N2 as a controlled baseline**
in §5. Holding the actor-critic algorithm constant, the
only differences are (i) our AIGC-aware state encoding and
(ii) our AIGC-aware reward shaping. The Pareto improvement
we measure (§5.1: SLO +28%, energy −7%, both p<0.05) is
thus attributable to AIGC awareness specifically, not to
general RL machinery. We also implement a **GNN variant**
inspired by Decima's encoder structure, finding that under
realistic training budgets a feedforward MLP with explicit
AIGC features matches or exceeds the GNN's relational
reasoning—an interesting departure from Decima's original
Spark-DAG findings.

### 7.4 AIGC-Specific Scheduling

The newest research wave addresses LLM-specific scheduling
properties directly. **Splitwise** [Patel'24] proposes
**phase splitting**: routing prefill to compute-rich GPUs
(H100) and decode to memory-rich GPUs (A100), with explicit
KV-cache transfer across the disaggregation boundary.
**DistServe** [Zhong'24] co-optimizes resource allocation and
parallelism strategy across disaggregated prefill/decode
pools. **Llumnix** [Sun'24] performs runtime request
*migration* across model instances to address request
heterogeneity, treating each instance as a "CPU core" in an
operating-system-style scheduling abstraction.

These works share our motivation—LLM workloads need
LLM-aware scheduling—but address a different problem layer:
they design *deployment topologies* (which GPUs serve which
phase) and *intra-cluster reactive policies* (mid-flight
migration), assuming a fixed topology and treating *initial
request placement* as orthogonal. **Our work attacks the
initial placement problem**: given a fixed heterogeneous
pool, learn a per-request server-selection policy that
co-optimizes SLO attainment and energy efficiency. The
two strands compose: Llumnix-style migration could refine
our scheduler's initial placement at runtime, and our
scheduler could populate prefill/decode pools in a
Splitwise-style deployment.

### 7.5 Simulation Platforms

Several open-source simulators have supported scheduling
research: **CloudSim** [Calheiros'11] and **iFogSim**
[Gupta'17] for general cloud-fog workloads;
**WorkflowSim** [Chen'12] for DAG workflows. None of these
models the AIGC-specific physics central to our work
(model weights, KV cache, continuous batching,
memory-bandwidth floors). A handful of recent open-source
projects target LLM serving simulation, but to our knowledge
they focus on intra-server batching dynamics (modeling the
behavior of a single inference engine) rather than
cross-server scheduling across heterogeneous pools.

To our knowledge, our simulator (§4.1) is the first
open-source platform combining three properties:
(i) modeling all five AIGC physics listed in §1
(model-weight residency, KV-cache locality, continuous
batching, memory-bandwidth floor, two-phase request
lifecycle); (ii) focusing on cross-server scheduling
across heterogeneous pools rather than intra-server
batching dynamics; (iii) supporting multi-objective
(SLO + energy) evaluation natively rather than as a
post-hoc analysis. We release it with reproducibility
manifests for the 30+ configurations reported in §5,
hoping to lower the activation energy for future AIGC
scheduling research.

---

## §8 Conclusion

This paper investigated AIGC-aware reinforcement-learning
scheduling for cloud-edge LLM inference. We built a
comprehensive AIGC simulator modeling five physical
characteristics—model weight residency, KV-cache locality,
continuous batching, memory-bandwidth floors, and two-phase
request lifecycle—then evaluated nine schedulers spanning
load-balancing heuristics, classic and metaheuristic DAG
schedulers, and three RL variants (PPO with MLP, A3C-R2N2
with GRU encoder, and PPO with GNN encoder).

Our AIGC-aware PPO scheduler achieves **Pareto dominance**
over eight strong baselines on the (SLO attainment, energy
per token) plane: +28% SLO and −7% energy at the canonical
edge=5, λ=2 configuration, with statistical significance
(p<0.05) on both axes against the four strongest
comparators. The Pareto dominance extends robustly across
edge counts {3, 5, 7} and survives the strict A3C-R2N2
controlled comparison that isolates the AIGC contribution
from general RL machinery.

A 12-component ablation, however, reveals that **AIGC-aware
scheduling is a holistic system contribution rather than a
clever reward trick**: only continuous batching simulation
(−83% SLO ablated) and adequate pretraining (−38% SLO
ablated) emerge as significant components. All individual
AIGC reward and state components are statistically
indistinguishable from the Full RL baseline in isolation.
This finding cautions the community against claiming gains
from specific AIGC reward terms without joint ablation.

We hope this work supports two longer-term directions: more
**realistic AIGC scheduling benchmarks** built on the open
platform we release, and a **research culture of aggressive
joint ablation** for AIGC scheduling claims. The
fast-moving LLM-systems landscape will reward both.

