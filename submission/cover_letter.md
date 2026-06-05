# Cover Letter — Future Generation Computer Systems

> Fill in the bracketed [...] fields before submitting. Paste the
> body into the Elsevier "Cover Letter" box (plain text), or attach
> as a PDF. Keep it to one page.

---

[Date]

To the Editor-in-Chief and Editorial Board
Future Generation Computer Systems
Elsevier

**Re: Submission of original research manuscript**
**Title:** "Pareto-Dominant Reinforcement Learning for Cloud-Edge
LLM Inference Scheduling"

Dear Editors,

We are pleased to submit our manuscript for consideration as a
regular research article in *Future Generation Computer Systems*.

Large language model (LLM) inference has become the
fastest-growing workload in modern data centers, and serving it
across heterogeneous cloud-edge pools is now a first-order systems
problem. Existing cluster schedulers were designed for generic DAG
workloads and do not model the physical characteristics that make
LLM inference distinctive—model-weight residency, KV-cache
locality, continuous batching, the memory-bandwidth floor of
decode, and the two-phase prefill/decode lifecycle. Our work
addresses this gap directly.

The manuscript makes four contributions:

1. **An open AIGC inference simulator** that models all five of the
   above physics as four engineering milestones, calibrated to vLLM
   and TensorRT-LLM measurements.
2. **An AIGC-aware PPO scheduler** that exposes these physics to the
   policy through state features and reward shaping, and achieves
   **strict Pareto dominance** over eight strong baselines
   (heuristic, load-balancing, and learned) on the
   (SLO attainment, energy-per-token) plane—+28% SLO and -7% energy
   per token at the canonical operating point, statistically
   significant at p<0.05.
3. **A 12-component ablation** with a deliberately reported null
   result: no single AIGC reward or state component is individually
   significant once continuous-batching simulation and pretraining
   are in place. We argue this is a methodological caution for the
   community—AIGC scheduling gains should be attributed to a
   holistic system, not to individual reward tricks, and verified by
   joint ablation.
4. **An empirical regime characterization** showing where AIGC-aware
   RL helps most (moderate load) and where it degrades to
   energy-only optimization (saturation, capacity-bound tiers).

We believe the work fits the scope of *FGCS* in distributed and
cloud computing, scheduling, and AI for systems, and that the
multi-objective (performance + energy) framing aligns with the
journal's emphasis on sustainable, future-generation
infrastructure. We release the simulator and reproducibility
manifests for all 30+ reported configurations.

**Originality and prior dissemination.** This manuscript is
original, has not been published elsewhere, and is not under
consideration by any other journal. All authors have approved the
submission and declare no conflict of interest.

**Suggested reviewers.** We suggest the following experts whose
work is closely related and who, to our knowledge, have no
conflict of interest with the author (no shared affiliation,
advising relationship, or recent co-authorship). Pick any three;
emails should be taken from each scholar's official institutional
page.

1. **Hao Zhang** — University of California, San Diego (CSE / HDSI).
   LLM serving systems and inference engines (vLLM, DistServe).
   Email: haz094@ucsd.edu
2. **Xin Jin** — Peking University (School of Computer Science).
   LLM serving and networked-systems scheduling (DistServe).
   Email: [VERIFY — found string was malformed; likely
   xinjin@pku.edu.cn or xinjinpku@pku.edu.cn, confirm at
   xinjin.github.io]
3. **Rajkumar Buyya** — University of Melbourne (CLOUDS Lab).
   Cloud–edge/fog scheduling and energy-aware resource management.
   Email: [listed publicly on http://www.buyya.com]
4. **Ricardo Bianchini** — Microsoft Research.
   Datacenter resource management and energy efficiency (Splitwise).
   Email: [on his Microsoft Research profile page]

We thank you for considering our submission and look forward to the
reviewers' feedback.

Sincerely,

Zhuolun Li (corresponding author)
Faculty of Science and Engineering, University of Bristol, Bristol, UK
Email: nu25406@bristol.ac.uk · ORCID: [fill in]
