# Extended Abstract

## Title

**EcoBudget: Task-Sufficient Web Retrieval for Greener 5G Question Answering**

## Author(s) and Affiliation(s)

Kasmya Bhatia, Amisha Singh, Ashwarya Singh
Department of Artificial Intelligence and Machine Learning, Manipal University Jaipur, Rajasthan, India
Contact: amificent21@gmail.com

## Purpose

Retrieval-augmented question answering (QA) systems typically load a fixed context
window or fetch as much of a web page as they can and let a language model sort out
what matters. Both strategies move far more data than any single question needs. On
mobile and 5G access networks this waste is not free: transferred bytes consume
radio energy and produce carbon, and each fetch holds the device modem in a
high-power state. The purpose of this work is to reframe retrieval as a problem of
*task sufficiency*, that is, to identify exactly what information a question
requires, retrieve only until the gathered evidence is sufficient, then stop, and to
measure rigorously whether the bytes and energy saved by loading less actually
outweigh the energy spent running the models. We build and evaluate EcoBudget, a
question answering pipeline that runs entirely on local models so that its
efficiency and energy claims reflect the deployed system rather than an external
service.

## Brief Literature Review

Retrieval-augmented generation grounds language models in retrieved passages, and a
growing line of work makes retrieval *adaptive* rather than fixed. Self-RAG learns
to decide when to retrieve and to critique its own generations, FLARE triggers
retrieval when generation confidence drops, and Adaptive-RAG routes a query by
predicted complexity to no, single, or multi-step retrieval. These methods optimise
answer quality, latency, or top-k relevance. Separately, the contextual-bandit
literature provides principled online decision policies, including LinUCB and linear
Thompson sampling, that balance exploration and exploitation with theoretical
guarantees. On the networking side, mobile energy is dominated by the radio access
network and by radio-state behaviour: measurement studies of 4G and 5G report that
the modem is held in a high-power connected state with an inactivity tail before it
releases to idle, so energy depends on how long and how often the radio is active,
not only on bytes moved. Web-sustainability research further estimates the carbon of
data transfer. What is not yet joined up is the intersection of these threads: an
adaptive retrieval policy whose explicit objective is transferred bytes and 5G radio
energy, evaluated on realistic page sizes with a defensible radio-state energy model.

## Research Gap

Existing adaptive-retrieval systems treat efficiency as latency or token budget, not
as transferred bytes and radio energy, and they rarely verify that the energy saved
by loading less exceeds the energy spent running the models. Energy studies of RAG,
where they exist, tend to assume unrealistically small payloads and ignore
radio-state tail energy. Conversely, networking energy models are seldom driven by a
task-level decision policy. The gap this work addresses is a task-sufficiency
stopping policy that is optimised for and evaluated on data transfer and 5G radio
energy, at measured page sizes, against a full complement of fixed-budget, full-page,
heuristic, and standard-bandit baselines, with the compute-versus-transfer trade-off
made explicit.

## Design/Methodology/Approach

EcoBudget decomposes a question into a small set of explicit information
requirements, each a structured (entity, attribute) pair, using a flan-t5-base model
fine-tuned with a LoRA adapter whose output attribute is snapped to a closed
35-attribute vocabulary (a form of constrained decoding). For each requirement, a
two-stage entity-aware retriever over a MiniLM-embedded corpus first gates candidates
to the requirement's entity and then ranks them by the attribute phrase, so the
correct fact is surfaced rather than the passage most similar to the entity name. An
evidence tracker built on a RoBERTa question-answering model scores whether each
requirement is satisfied, giving a sufficiency signal that raw embedding similarity
cannot. After each retrieval an adaptive stopping policy chooses STOP or RETRIEVE by
trading marginal evidence coverage against byte cost; the reported policy is a LinUCB
contextual bandit, with a coverage heuristic as a strong reference and a custom
online SGD bandit and linear Thompson sampling as further comparisons. A flan-t5-base
answerer generates the final answer from only the gathered evidence, once per
requirement.

Evaluation uses a hand-built and templated dataset of 186 labelled passages and
1,376 tasks (344 hand-authored seeds plus paraphrases) across seven answer types,
split at the seed level into frozen train, validation, and test partitions of 1,008,
240, and 128 tasks with a coverage invariant that prevents leakage. We pre-register
configurations, tune only on validation, and run the frozen test set exactly once.
Energy is charged with a first-order model that separates compute (model FLOPs
divided by device efficiency) from 5G transfer (a per-bit access-path model split
across radio access network, transport, and device modem) and a Radio Resource
Control (RRC) radio-state model with an inactivity tail; every coefficient is a cited
literature value with reported sensitivity ranges. Crucially, payload sizes are not
assumed: we measured the real rendered-HTML byte sizes of 19 pages fetched with a
browser engine (median 506 KB) and use that measured median for the page scenario.
Metrics are reported as binary judge success and continuous fact-level F1 with paired
bootstrap 95% confidence intervals, and we additionally run the full pipeline
end-to-end on live web pages.

## Findings (Expected/Preliminary)

All values below are measured on the frozen test set or on real pages unless noted.
The entity-aware retriever raised recall@1 from 0.914 to 0.989 and recall@5 from
0.983 to a perfect 1.000 on 175 unique requirements, eliminating a known
list-retrieval failure. On the frozen test set, task-sufficient adaptive stopping
reached 0.895 answer success and 0.944 fact-F1 using about 106 bytes per query,
versus 420 bytes for full-page loading, a reduction of roughly 75 percent at
statistically equal success, and it also beat naive one-passage-per-requirement
retrieval on success by 6.5 points. Against full-page loading the policy transferred
314 fewer bytes (95 percent CI [282, 350]) and, at the measured 506 KB page scale,
saved about 66 joules per query (95 percent CI [41, 95]); at that scale transfer
accounts for roughly 95 percent of total energy, whereas at snippet scale compute
dominates transfer by about 100 times. This compute-versus-transfer crossover is a
central result: loading fewer pages is the decisive energy lever precisely in the
realistic regime. On the 5G radio dimension, under aggressive fast-dormancy release
the policy used about 26 joules of radio energy per query versus about 107 for
full-page loading, roughly four times less, and the direction held across all 18
tested coefficient settings. Run end-to-end on live web pages, task-sufficient
loading cut transferred bytes by 92.8 and 97.8 percent, about 14 times less transfer
energy.

Two findings are reported candidly. First, once the strong retriever makes the first
result almost always correct, the learned policy converges to the hand-tuned
heuristic; its advantage over the heuristic grows with retrieval uncertainty. Across
five seeds the custom SGD bandit was unstable (0.790 plus or minus 0.283), while
LinUCB (0.913 plus or minus 0.007) and the heuristic (0.917) were stable, so LinUCB
is the reported learned policy and the SGD bandit is retained only as an ablation.
Second, on raw live-page text answer quality degrades, because the models were
trained on a clean structured corpus and live pages carry no attribute metadata; the
byte and energy savings, however, transfer to the real web unchanged.

## Research Implications

The work shows that efficiency in retrieval-augmented QA is better framed as an
energy-and-transfer objective than as a token budget, and that the honest measure of
"green" retrieval must net compute energy against transfer energy at realistic
payloads rather than assuming bytes saved are free. The compute-versus-transfer
crossover implies that conclusions about whether adaptive retrieval is green are
payload-scale-dependent, which reframes how such systems should be benchmarked.
Methodologically, the study demonstrates that a strong retriever can erase the
advantage of a learned stopping policy over a well-designed heuristic, so the value
of learning is conditional on residual uncertainty; this is a cautionary result for
claims that learned adaptive-retrieval controllers are intrinsically superior. The
finding that standard bandits (LinUCB) are more stable than a bespoke online policy
argues for reusing well-understood algorithms rather than novel controllers.

## Practical Implications

For deployed AI and language-model applications, capping retrieval by an evidence
sufficiency signal lowers serving cost and context waste without measurably hurting
answer quality. For mobile and edge networks, issuing fewer and earlier fetches
shortens radio active time and reduces RRC tail energy, which matters most on 5G
where the radio dominates energy; the effect is a concrete, if regime-dependent, per-
query saving. For sustainable computing, the measured per-query byte and energy
reductions compound at population scale into meaningful transfer and carbon savings.
For web and search systems, delivering task-relevant fragments instead of whole pages
directly cuts bandwidth for mobile clients. The reported pipeline is a practical
template: local models, a sufficiency-gated stopping policy, and per-query energy
accounting that a service can log and optimise.

## Originality/Value

The originality of EcoBudget lies in treating retrieval as task sufficiency with an
explicit 5G energy and radio-state objective, and in grounding that claim in
measured evidence rather than assumptions: real fetched page sizes, cited radio
coefficients with sensitivity analysis, a frozen single-shot test protocol, paired
bootstrap confidence intervals, and a live end-to-end demonstration on real web
pages. Equally valuable is the project's scientific honesty: it reports the
compute-versus-transfer crossover that undercuts naive byte-saving claims, the
convergence of the learned policy to a heuristic under a strong retriever, the seed
instability that led us to report LinUCB rather than a bespoke bandit, and the
corpus-to-web domain gap in answer quality. The contribution is therefore not a new
learning algorithm but a task-sufficiency framework and a defensible, reproducible
5G-green characterisation of when loading less genuinely pays off.

## Keywords

task-sufficient retrieval; adaptive stopping; 5G radio energy; retrieval-augmented
question answering; data-transfer efficiency; sustainable computing
