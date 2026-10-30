# Transformer Labs

This collection is the independent `transformer-labs` Git repository, checked out at `ai_projects/projects/transformer-labs/` as a submodule of `ai_projects`. Its project folders are ordinary directories tracked by this collection. Shared tutorials and learning checkpoints remain in the umbrella Docs submodule. Local bare origin: workspace `.ai/repositories/transformer-labs.git`.

Commit project changes in this collection first, then commit the updated collection pointer in `ai_projects`. Local commits are not automatically copied to the bare origin; run `git push origin main` in this collection before another checkout needs the new commit. No external hosting is configured.

I'm using these small projects to understand how text and image inputs become tokens, how embeddings and positions represent them, and how encoder and decoder blocks learn useful behavior.

Each lab is an ordinary folder in this collection repository and keeps its own Python package, environment, notebook and implementation notes when needed. Individual labs are not nested repositories or submodules.

| Lab | What I'm learning | Current state |
| --- | --- | --- |
| [Tokenization and Transformer Foundations](transformer-foundations-lab/README.md) | Numerical functions, gradients, tokenization and the path to a tiny encoder–decoder | Lesson 2 setup complete; `linear()` is next |
| [Attention Is All You Need](attention-is-all-you-need/README.md) | Attention, masking and English-to-French encoder–decoder translation | Attention implemented and checked; in-depth setup/attention tutorial available; data/model/training still pending |
| [Semantic Notes Search](semantic-notes-search/README.md) | Text embeddings, encoder fine-tuning and retrieval by meaning | Planned; detailed README only, with a launcher resource profile |
| [Tiny Vision Transformer](tiny-vision-transformer/README.md) | Image patches, positional embeddings and encoder classification | Planned; detailed README only, with a launcher resource profile |
| [Support Ticket Router](support-ticket-router/README.md) | Train embeddings and an intent classifier; compare word tokenization, BPE and subwords | Standalone boilerplate prepared; explicit source-file inputs; data, models and training start later |

I can add more bounded labs here as I need them: a causal decoder exercise or positional-encoding experiments. Support Ticket Router now holds the planned tokenizer/embedding comparisons inside an application; none of its models has been implemented or trained yet.

The later complete language-model builds belong in [Language Models](../language-models/README.md). Shared learning order and gates stay in the [roadmap](../../docs/guide/roadmap.md), with the actual next action in [resume](../../docs/guide/resume.md).

From workspace home, the existing launcher IDs still work:

```bash
./ai run transformer-foundations-lab
./ai run attention-is-all-you-need
```

To create another implementation lab later:

```bash
./ai new project my-lab --collection transformer-labs --title "My lab"
```

This creates an ordinary folder with a Python scaffold, not a completed model. The two planned encoder projects already have folders; continue them rather than recreating them.
