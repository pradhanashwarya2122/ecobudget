# EcoBudget: Task-Sufficient Web Retrieval with Adaptive Stopping for Data-Efficient 5G Access

**Abstract (approximately 500 words)**

Retrieval-augmented question answering typically loads a fixed context window or
ingests whole web pages and lets the model sort out what matters. Both strategies
move far more data than a task actually needs, and most transferred bytes never
affect the answer. On 5G access links this waste is not free: moving bytes consumes
radio energy, keeps the modem in a high-power state, and adds latency and carbon.
As data-intensive applications such as web search, AI agents, and edge services
scale, unnecessary application-layer transfer becomes a genuine efficiency problem.
EcoBudget reframes retrieval around task sufficiency. Rather than only asking "what
is relevant?", it asks "how much is enough?", retrieving only the evidence required
to complete a task and stopping once that evidence is sufficient.

EcoBudget decomposes a question into explicit (entity, attribute) information
requirements using a fine-tuned flan-t5-base model with a LoRA adapter, whose
output is snapped to a closed 35-attribute vocabulary. For each requirement it
performs entity-aware two-stage retrieval over a MiniLM-embedded corpus, gating
candidates to the correct entity before ranking them by attribute. An evidence
tracker built on a RoBERTa question-answering confidence signal judges whether each
requirement is satisfied, and an adaptive stopping controller decides at each step
whether to STOP or RETRIEVE by weighing marginal evidence coverage against data
cost. The answer is generated only from the evidence actually gathered. The entire
pipeline runs on local models with no hosted LLM API, so efficiency claims reflect
the deployed system, and evaluation uses a frozen train, validation, and test
split with bootstrap confidence intervals and a single final test-set run.

Measured results support the central hypothesis. Entity-aware retrieval raises
Recall@1 from 0.914 to 0.989 and Recall@5 from 0.983 to 1.000. The fine-tuned
decomposer reaches 0.892 exact-match, with constrained snapping to the attribute
vocabulary lifting it from 0.496. On the frozen test set, adaptive stopping holds
0.895 task success while using about 106 bytes per query versus 420 bytes for
full-page loading, roughly 75 percent fewer bytes at statistically equivalent
success. Using a measured median real page of 506 KB, adaptive stopping saves about
66 joules per query versus full-page loading (95 percent confidence interval 40.8
to 95.1), and end-to-end on live web pages it transferred 92.8 and 97.8 percent
fewer bytes. Because 5G energy is dominated by keeping the radio active, fewer and
earlier fetches also reduce radio and tail energy, about four times lower than full
loading under aggressive dormancy.

The findings recast web retrieval as a task-level resource-allocation problem
rather than pure relevance ranking. Requirement-aware retrieval surfaces the right
fact instead of the most entity-similar text, and adaptive stopping avoids retrieval
that no longer changes the answer. With a strong retriever the learned controller
matches a hand-tuned heuristic and its advantage grows with retrieval uncertainty;
LinUCB is the stable reported controller. EcoBudget offers a data-efficient
retrieval framework and a pathway toward sustainable 5G access; broader energy and
carbon benefits require direct hardware measurement in future work.
