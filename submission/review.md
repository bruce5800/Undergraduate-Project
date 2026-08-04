Reviewer #1: The proposed work addresses an interesting and timely problem, and I agree that the research topic is both relevant and significant. However, the manuscript still lacks several important aspects, and significant revisions are required before it can be considered suitable for publication in this journal.More detailed comments are provided below:

- Several acronyms lacks of proper definition before usage: AIGC, GA, PSO, DAG, HPC, and several others. Please ensure that every acronym is introduced at its first occurrence and used consistently throughout the manuscript.

- Both the abstract and the introduction refer to "collaborative LLM inference scheduling." However, it is not clear what 'collaborative' exactly means in the context of this work. From my understanding, all the approaches included in the experimental evaluation rely on a single-agent based scheduling mechanism and do not involve any explicit collaborations. Please clarify this terminology or justify its use.

- In the introduction it is stated that the scheduling approach studied in this work is gaining traction for three main reasons: costs, privacy, and tail latency. However, among these motivations, only latency appears to be modeled in the optimization objectives through the SLO formulation. The manuscript should either incorporate the remaining aspects into the optimization framework or clarify why they are discussed as motivations but not considered in the proposed formulation.

- The five physical characteristics that define LLM inference tasks are well introduced and effectively motivate this work. However, the equations, parameter values, and assumptions presented in Section I.A are provided without references to prior studies or supporting sources.

- A Table gathering and describing all the notations used in section III would greatly improve the readability of the manuscript.

- In the request workload formulation, it is unclear what the target model represents. How is the workload able to determine which available model best matches its requirements? Please clarify this aspect.

- The reward is formulated as a weighted sum including seven different components, approximating the multi-objective formulation into a single-objective one. However, the experimental evaluation considers only one weight configuration. Consequently, the conclusions drawn cannot be considered fully general, as different weight configurations could potentially lead to different behaviors and trade-offs. I recommend evaluating multiple weight configurations and discussing their impact on the results. Additionally, the rationale behind the selected weights should be clearly explained and justified.

- While the manuscript introduces the selected baseline methods, it does not specify whether they correspond to existing implementations (e.g., open-source libraries) or custom implementations developed by the authors. Furthermore, important experimental details are missing, including hyperparameter configuration, number of RL training steps, or number of optimization iterations for the metaheuristics.

- Several tables presented in the experimental evaluation are never explicitly referenced in the text, making it difficult to associate the discussion with the corresponding results. Furthemore, Table I shows that the proposed RL performs best in every metric considered. However, considering the Mksp metric, A3C-R2N2 seems to perform better than the proposed method. Please verify and clarify this point.

- As noted above, several conclusions presented in Section VI are expressed as strong general claims. Given that the evaluation relies on a single reward weight configuration, I believe these claims should be supported through a more comprehensive sensitivity analysis.

- In the related work, references related to cluster scheduling state-of-the-art are old and quite generic (the most recent is from more than 20 years ago). Consequently, the manuscript does not adequately represent the current state of the art, particularly recent works on metaheuristic-based and other modern scheduling approaches. I encourage the authors to substantially update this section with more recent literature.

Reviewer #2: The Article is not properly uploaded. The author has uploaded latex file, which was not in readable format.

Reviewer #3: The manuscript studies a blockchain trusted collaboration framework for circular supply chains, but it contains irreparable critical issues, and the format is really chaotic. The rejection is suggested.

1. Novelty is extremely weak. The core modules merely combine existing technologies, without distinct theoretical innovations or fair comparison with state-of-the-art methods.

2. Experimental design is incomplete. Recent advanced baselines are absent; large-scale industrial verification and systematic bottleneck analysis are not provided.

3. Security analysis is superficial. Collusion and cross-chain tampering attacks are insufficiently discussed, lacking complete fault-tolerance recovery mechanisms.

4. Poor engineering practicability. Only simple simulation results are given, without operable deployment plans for real supply chain businesses.

5. Manuscript formatting is chaotic. The author did not carefully check the typesetting; numerous LaTeX compilation errors and garbled characters appear in the source file, seriously affecting reading.
