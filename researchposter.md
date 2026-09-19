# EcoBudget — Academic Poster Content

> All quantitative values below are **measured** in this project unless a line is
> explicitly labelled *(interpretation)* or *(implication)*. Items needing
> external confirmation are marked **[verify]**. No results are invented.

---

## Title

**EcoBudget: Task-Sufficient Web Retrieval for Greener 5G Question Answering**

*Subtitle:* Deciding how much to load, per question, to cut data transfer and
energy without losing answer quality.

## Author(s) and Affiliation

**Kasmya Bhatia, Amisha Singh, Ashwarya Singh**
Department of Artificial Intelligence and Machine Learning, Manipal University Jaipur, Rajasthan, India
Contact: amificent21@gmail.com

---

## 01 — Introduction & Objectives

**Introduction / Background** (≈110 words)
Retrieval-augmented question answering usually loads a fixed context window or
grabs as much of a web page as possible and lets the model sort it out. Both
waste data: most transferred bytes never affect the answer. On mobile and 5G
links this waste is not free — moving bytes consumes radio energy and produces
carbon, and the radio is held in a high-power state while data flows. EcoBudget
reframes retrieval as *task sufficiency*: identify exactly what a question needs,
retrieve only until the evidence is sufficient, then stop. The system runs
entirely on local models (no hosted LLM API), so efficiency claims reflect the
deployed pipeline rather than an external service, making the energy accounting
honest and reproducible.

**Objectives** (5 bullets, 10–15 words each)
- Decompose questions into explicit (entity, attribute) information requirements.
- Retrieve evidence per requirement and rank candidates by task usefulness.
- Learn an adaptive stopping policy that halts retrieval once evidence suffices.
- Quantify bytes, compute energy, and 5G radio energy saved per query.
- Compare adaptive stopping against fixed-budget and full-page loading baselines.

---

## 02 — Research Gap, Question & Hypothesis

**Research Gap** (≈70 words)
Existing retrieval and adaptive-RAG systems optimise answer quality, latency, or
top-k relevance, but rarely treat *transferred bytes and radio energy* as
first-class objectives, and rarely verify that the energy saved by loading less
exceeds the energy spent running the models. They also assume unrealistically
small payloads. What is missing is a task-sufficiency stopping policy evaluated on
realistic page sizes with a defensible, 5G-grounded energy and radio-state model.

**Research Question**
Can a per-question adaptive stopping policy reduce data transferred and 5G energy
compared with fixed-budget and full-page retrieval, while maintaining answer
success?

**Hypothesis**
Retrieving only until evidence is sufficient will transfer substantially fewer
bytes and less radio energy than fixed or full-page loading, at statistically
equivalent answer success.

---

## 03 — EcoBudget Approach

**Approach** (≈110 words)
EcoBudget converts a question into a small set of explicit information
requirements, each a structured (entity, attribute) pair. It retrieves evidence
for each requirement separately, gating candidate passages to the correct entity
before ranking them by the attribute so the right fact is surfaced rather than the
most entity-similar text. After each retrieval, an evidence tracker scores whether
the requirement is satisfied using a question-answering confidence signal rather
than raw similarity. A learned stopping policy then chooses STOP or RETRIEVE by
trading marginal evidence coverage against byte cost. The answer is generated only
from the evidence actually gathered, and every query logs bytes moved, compute
energy, and 5G radio energy so efficiency is measured, not assumed.

**Process steps** (6 numbered, 10–20 words each)
1. Receive the user question and task type.
2. Decompose it into (entity, attribute) information requirements.
3. Retrieve candidate passages per requirement: gate to entity, rank by attribute.
4. Score evidence sufficiency per requirement using QA confidence.
5. Adaptive policy decides STOP or RETRIEVE from coverage versus byte cost.
6. Generate the answer from gathered evidence; log bytes and energy.

---

## 04 — System Architecture & Methodology

**Methodology** (≈140 words)
A user question and task type enter the pipeline. A fine-tuned flan-t5-base (with
a LoRA adapter) decomposes it into (entity, attribute) requirements, snapped to a
closed 35-attribute vocabulary for reliability. For each requirement, an
entity-aware two-stage retriever over a MiniLM-embedded corpus performs web/search
retrieval: it first restricts candidates to the requirement's entity, then ranks
by attribute similarity. An evidence tracker built on a RoBERTa QA model evaluates
whether each requirement is satisfied, giving a per-requirement
sufficiency/confidence assessment that raw cosine similarity cannot. An adaptive
stopping mechanism (a LinUCB contextual bandit, with a coverage heuristic as a
strong reference) halts retrieval once evidence is sufficient. A flan-t5-base
answerer then generates the final
answer from only the gathered evidence. Experimental evaluation uses a frozen
train/validation/test split, bootstrap confidence intervals, pre-registration, and
a single final test-set run; energy is charged with a compute + 5G-per-bit + RRC
radio model on measured page sizes.

**Architecture flow (for diagram, 8 steps)**
1. Question + task input →
2. Decompose into (entity, attribute) requirements →
3. Per-requirement retrieval (entity gate → attribute rank) →
4. Evidence tracker: QA-confidence sufficiency check →
5. Adaptive policy: STOP or RETRIEVE? (loop back to 3 if RETRIEVE) →
6. Answer generation from gathered evidence →
7. Measure bytes + compute + 5G radio energy →
8. Report success, data transferred, energy.

---

## 05 — 5G Relevance

(≈70 words)
5G access energy is dominated by the radio: bytes moved consume per-bit energy,
and each fetch keeps the modem in the high-power RRC_CONNECTED state, incurring a
tail before it returns to idle. At realistic web-page sizes, transfer dominates
total per-query energy. A policy that loads fewer pages and stops earlier
therefore reduces both bytes and radio active time — the levers that matter most
on data-intensive mobile and edge applications.

**Impact points**
- **Reduced data transfer:** fewer passages fetched means fewer bytes over the air.
- **Network efficiency:** fewer, earlier fetches shorten radio active time and tail energy.
- **Application-layer efficiency:** task-sufficient loading avoids full-page downloads for single facts.
- **Sustainability:** fewer transferred bytes and inferences lower per-query energy and carbon.

---

## 06 — Main Results

*(All values measured on the frozen test set / real pages unless noted.)*
- **Retrieval accuracy:** entity-aware retrieval raised recall@1 from **0.914 to 0.989** and recall@5 from **0.983 to 1.000** on 175 unique requirements.
- **Answer success held while bytes fell:** on the frozen test set adaptive stopping reached **0.895 success / 0.944 fact-F1** using **106 bytes/query** versus **420 bytes** for full-page loading (**~75% fewer bytes**) at statistically equal success.
- **Real live web pages:** end-to-end on live pages, task-sufficient loading cut transferred bytes by **92.8% and 97.8%** (≈14× less transfer energy).
- **Energy at realistic page scale:** with the measured median page (**506 KB**), on the frozen test set adaptive stopping saved **≈66 J/query versus full-page loading** (95% CI [40.8, 95.1]); transfer was about **95%** of total energy at this scale.
- **5G radio energy:** under aggressive fast-dormancy release, adaptive stopping used **≈26 J** of radio energy per query versus **≈107 J** for full-page loading (about **4× less**); the direction holds across all 18 tested coefficient settings.
- **Stable learned policy:** LinUCB is the reported and deployed stopping policy (val multi-seed **0.913 ± 0.007**); the custom SGD bandit is seed-unstable (**0.790 ± 0.283**) and retained only as an ablation.

---

## 07 — Supporting Results

**Graph 1 — Success vs Energy (Pareto frontier).**
Measures answer success against per-query energy at real page scale for every
condition. *Interpretation:* adaptive methods sit at the low-energy, high-success
corner while fixed and full-page loading spread out at far higher energy. *Why it
matters:* it shows task-sufficient loading is Pareto-optimal, not a quality trade.

**Graph 2 — Compute vs 5G transfer crossover.**
Measures compute energy versus transfer energy across payload realism (text →
full page). *Interpretation:* compute dominates at snippet scale (~100×) but
transfer overtakes it at real page scale. *Why it matters:* it justifies loading
fewer bytes precisely where deployments actually operate.

**Graph 3 — 5G RRC radio / tail energy.**
Measures per-query radio energy by condition under tight-loop and fast-dormancy
release. *Interpretation:* fewer fetches sharply cut radio energy, with the gap
widening to about 4× under aggressive release. *Why it matters:* it captures a 5G
cost that pure byte counts miss.

**Graph 4 — Measured byte savings on real pages.**
Measures full-page bytes versus task-sufficient bytes on live fetched pages.
*Interpretation:* task-sufficient loading transfers 90%+ fewer bytes on real web
content. *Why it matters:* it demonstrates the thesis on real pages, not only the
curated corpus.

---

## 08 — Discussion & Key Findings

**Discussion** (≈145 words)
The results support the hypothesis: task-sufficient stopping matches full-page
answer success while transferring far fewer bytes and less radio energy. The
behaviour follows from where evidence lives — most requirements are satisfied by
one or two passages, so continuing to retrieve adds bytes and radio active time
without raising coverage. Adaptive stopping exploits this by halting once the
sufficiency signal is met, which is why byte and energy savings are large while
success is unchanged. *(Interpretation)* Notably, once the entity-aware retriever
made the first result almost always correct, the learned policy converged to a
strong hand-tuned heuristic; the advantage of learning grows with retrieval
uncertainty. Across seeds the custom SGD bandit proved unstable, so the reported
learned policy is LinUCB, which is stable and matches the heuristic while
dominating fixed budgets. *(Limitation)* Answer quality degrades on raw, messy live-page text
because models were trained on a structured corpus, and the corpus spans limited
domains. Relative to the research question, the efficiency gains are confirmed and
measured; the learning-specific gain is conditional on uncertainty.

**Key Findings** (5, 15–25 words each)
- Task-sufficient stopping cut bytes ~75% versus full-page loading at equal test success (0.895).
- Entity-aware retrieval lifted recall@1 to 0.989 and recall@5 to a perfect 1.000.
- At realistic 506 KB pages, transfer dominates energy; on test, stopping early saved ~66 J/query versus full load.
- Fewer fetches reduced 5G radio and tail energy, with the largest gap (about 4×) under fast-dormancy release.
- A learned policy's edge over a strong heuristic depends on retrieval uncertainty; LinUCB (stable) is the reported policy.

---

## 09 — Conclusion & Practical Implications

**Conclusion** (≈90 words)
Retrieval-augmented QA wastes data by loading far more than each question needs,
and on 5G that waste costs radio energy and carbon. EcoBudget addresses this with
task-sufficient retrieval: decompose into information requirements, retrieve per
requirement, and stop once evidence is sufficient. Measured on a frozen test set
and on real web pages, it maintained answer success (0.895) while transferring
about 75% fewer bytes than full-page loading, saving roughly 66 J/query at
realistic page scale. The contribution is the task-sufficiency framework plus a
5G-grounded energy and radio-state characterisation of when loading less pays off.

**Practical Implications** (4 bullets, 15–25 words each)
- **AI/LLM applications:** cap retrieval by evidence sufficiency to lower serving cost and context waste without hurting answer quality.
- **Network efficiency:** fewer, earlier fetches shorten radio active time and reduce tail energy on 5G access links.
- **Sustainable computing:** measured per-query byte and energy reductions compound at population scale into meaningful carbon savings.
- **Web/search systems:** deliver task-relevant fragments instead of full pages to cut bandwidth for mobile and edge clients.

---

## 10 — References (APA 7th Edition)

> Real works are cited; uncertain specifics (full author lists, pages, DOIs,
> exact years/versions) are flagged **[verify]** rather than fabricated.

Asai, A., Wu, Z., Wang, Y., Sil, A., & Hajishirzi, H. (2023). *Self-RAG: Learning to retrieve, generate, and critique through self-reflection* [Conference paper]. **[venue/year — verify: ICLR 2024]**

Huang, J., Qian, F., Gerber, A., Mao, Z. M., Sen, S., & Spatscheck, O. (2012). A close examination of performance and power characteristics of 4G LTE networks. In *Proceedings of the 10th International Conference on Mobile Systems, Applications, and Services (MobiSys)* (pp. 225–238). Association for Computing Machinery. **[pages/DOI — verify]**

Jeong, S., Baek, J., Cho, S., Hwang, S. J., & Park, J. C. (2024). Adaptive-RAG: Learning to adapt retrieval-augmented large language models through question complexity. In *Proceedings of NAACL 2024*. Association for Computational Linguistics. **[pages/DOI — verify]**

Li, L., Chu, W., Langford, J., & Schapire, R. E. (2010). A contextual-bandit approach to personalized news article recommendation. In *Proceedings of the 19th International Conference on World Wide Web (WWW)* (pp. 661–670). Association for Computing Machinery. **[DOI — verify]**

Narayanan, A., et al. (2021). A variegated look at 5G in the wild: Performance, power, and QoE implications. In *Proceedings of the ACM SIGCOMM 2021 Conference* (pp. 610–625). Association for Computing Machinery. **[full author list & DOI — verify; APA 7 lists up to 20 authors]**

3rd Generation Partnership Project. (2023). *NR; Radio Resource Control (RRC) protocol specification* (3GPP TS 38.331, Release 17). **[release/year — verify]**

---

## Requirements checklist

- Title ✓
- Authors & Affiliation ✓ *(Kasmya Bhatia, Amisha Singh, Ashwarya Singh; Manipal University Jaipur)*
- Introduction ✓
- Objectives ✓
- Methodology ✓
- Key Findings/Results ✓
- Discussion ✓
- Conclusion ✓
- Practical Implications ✓
- APA 7 References ✓ *(uncertain specifics flagged [verify], none fabricated)*
