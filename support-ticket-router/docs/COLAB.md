# Colab preparation

The Colab notebook and Free profile are templates. Data preparation, dependencies for model work, training and hosted recovery are pending.

When I start, I'll use the [workspace Colab contract](../../../../docs/guide/COLAB.md) and the same Python trainer locally and in the notebook.

1. Transfer a versioned source package and verify source/config identity.
2. Pin the actual compatible Python/PyTorch/tokenizer dependencies. The current scaffold has no ML dependencies.
3. Acquire Banking77 reproducibly and freeze its train/validation/test manifest.
4. Measure a CPU smoke run and then the selected hosted hardware's memory and step time.
5. Select a bounded Free training budget from that evidence.
6. Save full model/optimizer/scheduler/scaler/progress/RNG/data-order state and source/config/tokenizer/data/environment identities to verified Drive generations, with an older valid fallback.
7. Delete the runtime, restore the same run and verify restored/advanced updates.

`configs/colab-free.json` is a proposed configuration, not a measured fit. Its download and training flags remain false. Changing those flags does not implement or enable any training command.

Runtime-local caches, persistent checkpoints and the best validation artifact will be separate. No dataset, model or Drive upload has occurred during boilerplate preparation.
