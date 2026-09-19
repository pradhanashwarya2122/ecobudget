# EcoBudget Poster: FINAL PASTE-READY CONTENT (codebase-authoritative)

All numbers are the current reproducible measurements from this project. No
placeholders, no invented values. No em dashes.

## TITLE
EcoBudget: Task-Sufficient Web Retrieval with Adaptive Stopping for Data-Efficient 5G Access

## AUTHORS
Kasmya Bhatia, Ashwarya Pradhan, Amisha Singh

## AFFILIATION
Manipal University Jaipur

## CONFERENCE
IC5G 2026, International Conference on 5G for Sustainable Development

## 01 INTRODUCTION
Web retrieval often transfers far more data than a task actually requires. In 5G,
data-intensive applications like web search, AI agents, and edge services make
this excess a real cost: more bytes moved, slower responses, and higher energy
demand. EcoBudget reframes the question. Instead of only asking "what is
relevant?", it asks "how much is enough?", delivering just the evidence needed to
finish the task efficiently.

## OBJECTIVES
- Retrieve only enough evidence to complete a task
- Avoid unnecessary application-layer data transfer
- Maintain task success while reducing retrieval
- Adapt retrieval effort to task difficulty

## 02 RESEARCH GAP
Existing retrieval systems optimize relevance, ranking, or fixed retrieval limits.
They rarely treat the data cost of retrieval as a first-class objective, and rarely
check whether one more retrieval step is worth its bytes. EcoBudget frames
retrieval as a sequential decision problem: at each step, decide whether more
evidence justifies its additional data cost.

## RESEARCH QUESTION
Can an adaptive controller learn when additional information is no longer worth its
data cost?

## HYPOTHESIS
An adaptive retrieval policy can reduce unnecessary data transfer while maintaining
task success; overly aggressive stopping, however, would reduce success.

## 03 ECOBUDGET APPROACH
Core idea: retrieve only the evidence needed to complete the task, and stop when
extra retrieval adds no value.
1. Understand the task: identify the information requirements needed to answer it.
2. Collect evidence: retrieve targeted to those requirements, not broadly.
3. Measure sufficiency: track evidence coverage and confidence against data already transferred.
4. Decide: STOP if evidence is sufficient; RETRIEVE if more is worthwhile.
5. Balance success and cost: maximize task success, minimize unnecessary data transfer.

## 04 SYSTEM ARCHITECTURE & METHODOLOGY (diagram labels)
INPUT: user question and task type.
1. Decompose into (entity, attribute) information requirements (fine-tuned
   flan-t5-base + LoRA, snapped to a closed 35-attribute vocabulary).
2. Retrieve per requirement (MiniLM bi-encoder; entity-aware two-stage: gate to
   entity, then rank by attribute).
3. Evidence sufficiency (RoBERTa QA-confidence per requirement, not raw similarity).
4. Adaptive stopping controller: STOP or RETRIEVE (loop back if RETRIEVE).
5. Answer generation from only the gathered evidence.
OUTPUT: answer plus measured bytes and energy.

## 05 5G RELEVANCE
5G enables high-throughput, data-intensive applications, and at scale their
unnecessary application-layer transfer becomes an efficiency problem. Less
unnecessary retrieval leads to less application-layer data transferred, which lowers
the network traffic a task generates, which yields potential efficiency and energy
benefits. EcoBudget acts on the first link in that chain: the bytes an application
requests.
Application areas: Sustainable Web, 5G / Edge Networks, AI Agents & Search.

## 06 MAIN RESULTS (codebase-authoritative)
- Requirement-aware retrieval: entity-aware retrieval raises Recall@1 from 0.914
  to 0.989 (and Recall@5 from 0.983 to 1.000) over 175 unique requirements.
- Task decomposition: the fine-tuned decomposer reaches 0.892 exact-match on the
  validation split; constrained snapping to the attribute vocabulary lifts it from
  0.496.
- Adaptive stopping: on the frozen test set it holds 0.895 task success while using
  about 106 bytes per query versus 420 bytes for full-page loading, about 75 percent
  fewer bytes at statistically equal success.
- Energy at realistic page scale: with the measured median page of 506 KB, adaptive
  stopping saves about 66 J per query versus full-page loading (95% CI 40.8 to 95.1).
- Real live web pages: end-to-end on live pages, task-sufficient loading cut
  transferred bytes by 92.8 percent and 97.8 percent.

## 07 SUPPORTING RESULTS (codebase-authoritative)
- Snap-to-vocabulary ablation: constrained decoding onto the 35-attribute vocabulary
  raises decomposer exact-match from 0.496 to 0.892 (attribute-only).
- Policy stability: LinUCB is the deployed controller, stable across seeds (0.913 +/-
  0.007); the custom SGD bandit is seed-unstable (0.790 +/- 0.283) and kept only as
  an ablation.
- Compute vs transfer: compute dominates by about 100x at snippet scale, but transfer
  becomes about 95 percent of per-query energy at realistic page scale; 5G radio
  energy under fast-dormancy is about 4x lower for adaptive stopping than full load.

## 08 KEY FINDINGS
1. Requirement-Aware Retrieval: targeting individual information requirements rather
   than the whole query improves retrieved-evidence quality, raising Recall@1 from
   0.914 to 0.989 and Recall@5 to 1.000.
2. Task Decomposition: a fine-tuned (entity, attribute) decomposer reaches 0.892
   exact-match, with constrained snap-to-vocabulary adding about 0.40 over the
   un-snapped model.
3. Adaptive Stopping: balancing evidence sufficiency against retrieval cost holds
   full-page task success (0.895) at about 75 percent fewer bytes and about 66 J per
   query less energy, dominating fixed-budget baselines.

## DISCUSSION
EcoBudget shows that web retrieval can be treated as a task-level resource-allocation
problem, not only a relevance-ranking problem. Requirement-aware retrieval improves
the quality of retrieved evidence, because targeting a specific (entity, attribute)
surfaces the right fact instead of the most entity-similar text. Adaptive stopping
provides the mechanism to avoid retrieval that no longer changes the answer: most
requirements are satisfied by one or two passages, so further retrieval adds data
cost without adding coverage. With a strong retriever the learned controller matches
a hand-tuned heuristic; its advantage grows as retrieval uncertainty rises. The
relationship between evidence sufficiency and retrieval cost is what the controller
learns to exploit.

## 09 CONCLUSION
EcoBudget treats retrieval as deciding how much to load, not just what is relevant.
Requirement-aware retrieval improves evidence quality, and adaptive stopping halts
once evidence is sufficient, trading task success against data-transfer cost. On a
frozen test set it holds 0.895 success at about 75 percent fewer bytes than full-page
loading. The contribution is a task-sufficiency framework for data-efficient
retrieval, a step toward sustainable 5G access. Retrieve enough; stop when enough is
enough.

## PRACTICAL IMPLICATIONS
- AI applications: retrieve according to what the task needs, not how much
  information is available.
- Network efficiency: reducing unnecessary application-layer transfer lowers the data
  retrieval-intensive apps request from the network.
- Sustainable computing: selective retrieval is a pathway toward data-efficient AI;
  broader energy and carbon benefits require direct hardware measurement in future work.

## 10 REFERENCES (APA 7)
- Chung, H. W., et al. (2022). Scaling instruction-finetuned language models. arXiv.
  [full author list and venue: verify]
- Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen,
  W. (2022). LoRA: Low-rank adaptation of large language models. International
  Conference on Learning Representations (ICLR). [verify year]
- Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese
  BERT-networks. Proceedings of EMNLP-IJCNLP 2019. [pages/DOI: verify]
- Li, L., Chu, W., Langford, J., & Schapire, R. E. (2010). A contextual-bandit approach
  to personalized news article recommendation. Proceedings of WWW 2010, 661-670.
  [DOI: verify]  (add only if you cite the LinUCB controller)

## QR / DEMO / GITHUB / CONTACT
[NOT PROVIDED IN RESEARCH]. Add your own repository URL, demo link, QR destination,
and contact email. Do not invent one.
