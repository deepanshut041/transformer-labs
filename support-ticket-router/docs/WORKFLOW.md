# Support Ticket Router workflow

My goal is to **train embeddings and an intent classifier; compare word tokenization, BPE and subwords** in a practical routing application.

The [phase overview](../phase.md) is the ordered route. [Detailed phase plans](../plan/00-resume.md) are local Markdown files ignored by Git. Each phase states its requirements, inputs, outputs, review chunks and completion checks before coding.

For each phase: explain the requirements and a worked whiteboard example, have the assistant write the corresponding code/tests, then review the implementation and results together. Use one coherent explanation or multiple chunks depending on the phase. The first phase has two chunks: trusted raw records, then stable intent targets.

Only the boilerplate is ready. Every learning stage below still needs implementation. Standalone scripts declare their inputs in their `if __name__ == "__main__":` blocks and report an explicit deferred-stage error until implementation starts.

| Stage | Next implementation | Evidence later |
| --- | --- | --- |
| Dataset and splits | Pin Banking77, inspect requests, freeze splits | Provenance, attribution and leakage checks |
| Word / BPE / subwords | Train tokenizers only on training text | Deterministic save/load and input coverage |
| Embeddings | Joint learning, Word2Vec and character n-grams | Numerical checks and actual trained vectors |
| Intent classifier | Masked pooling, MLP and cross-entropy | Gradient checks and tiny-set overfit |
| Colab training | Common trainer with full-state persistence | Measured fit and recovery after runtime deletion |
| Comparison | Same splits and recorded budgets | Held-out macro-F1/accuracy and concrete mistakes |
| Application | Category suggestions, similar examples, corrections | A usable demo backed by trained artifacts |

BPE is a subword tokenizer. The additional subword comparison is Unigram; character n-grams belong to the separate embedding comparison. Keep tokenizer choices and embedding methods distinct so a measured difference can be interpreted.

Banking77 provides intent-labelled requests. Similar examples won't be described as resolved tickets without resolution data. Corrections from the application must be reviewed before entering a future training set.

**Next actual project session:** follow [Phase 01 requirements](../plan/phase-01-data-contract.md), explain the raw-row contract, then implement and review acquisition/loading/validation before the label-map chunk. Splitting and the lexical baseline are Phase 2. No phase implementation has started.
