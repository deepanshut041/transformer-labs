# Sources

I'm using Priyam Mazumdar's three videos for this build. The sequence starts with attention, moves to English/French data preparation, then puts the Transformer and training loop together.

| Part | Video | Topics | Chapter starts |
| --- | --- | --- | --- |
| 1 | [Attention is All You Need: Ditching Recurrence for Good!](https://www.youtube.com/watch?v=JXY5CmiK3LI) | Attention projections and heads, padding and causal masks, cross-attention, optimized attention | 00:25:00 initial implementation; 01:09:20 packed multi-head attention; 01:26:30 padding; 02:03:08 causal; 02:17:30 cross-attention |
| 2 | [Prepping Data for Neural Machine Translation](https://www.youtube.com/watch?v=UkBggwQsbF8) | WMT English/French pairs, French WordPiece, English BERT tokenizer, shifted targets, collation | 00:02:00 tokenizer; 00:30:00 dataset; 00:41:35 tokenization; 00:56:00 training format; 01:08:35 collator |
| 3 | [Build + Train the Transformer for Neural Machine Translation!](https://www.youtube.com/watch?v=DOqg_QxInd8) | Positional/token embeddings, attention, FFN, encoder, decoder, inference and training | 00:11:40 positions; 01:01:36 encoder; 01:09:30 decoder; 01:21:10 assembly; 01:35:23 inference; 01:54:38 training |

## Paper and reference code

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Attention code](https://github.com/priyammaz/PyTorch-Adventures/tree/2f18a5470c4ef0a3d413dd7b18d07941f45779e9/PyTorch%20for%20Transformers/Attention%20Mechanisms/Attention)
- [Translation code](https://github.com/priyammaz/PyTorch-Adventures/tree/2f18a5470c4ef0a3d413dd7b18d07941f45779e9/PyTorch%20for%20NLP/Seq2Seq%20for%20Neural%20Machine%20Translation)
- [WMT14 translation task](https://www.statmt.org/wmt14/translation-task.html)

The reference is [PyTorch-Adventures](https://github.com/priyammaz/PyTorch-Adventures), at commit `2f18a5470c4ef0a3d413dd7b18d07941f45779e9`, under the MIT license. The repository tree and the attention/translation READMEs were checked during setup on 2026-10-10. The attention implementation adapts the reference notebook under its MIT license; no upstream tokenizers or weights have been copied.

I'll keep the videos' separate English and French tokenizers and record any changes I make to the reference settings. The CPU and Colab configs are small starting points. Before using WMT data, I still need to choose a subset and splits and check the license. The full corpus hasn't been downloaded. The video 1 implementation includes the efficient SDPA path. Large or distributed training can wait until the small model works.

## Captions

English `en-orig` and `en` auto-captions were downloaded for all three videos on 2026-10-10. Titles, dates, chapters, file sizes and SHA-256 hashes are in [sources.json](sources.json). Auto-captions can get technical terms wrong, so I use them alongside the video chapters and reference READMEs.

The original captions and timestamped reading copies are in the ignored `.sources/subtitles/` directory. The initial download is also saved under workspace `.ai/video-sources/2026-10-10-attention-is-all-you-need/`. Captions stay out of Git and the Colab source ZIP. No video or audio files were downloaded.

To download the captions again from this project's directory, use this argument array:

```text
["uvx", "--from", "yt-dlp==2026.08.19", "yt-dlp", "--skip-download", "--write-subs", "--write-auto-subs", "--sub-langs", "en.*", "--sub-format", "vtt", "--no-playlist", "-o", ".sources/subtitles/%(id)s.%(ext)s", "https://www.youtube.com/watch?v=JXY5CmiK3LI", "https://www.youtube.com/watch?v=UkBggwQsbF8", "https://www.youtube.com/watch?v=DOqg_QxInd8"]
```

The hashes identify the saved copies. A later download may differ if the captions change.

## Video 1 implementation

I'm turning the reference notebook into importable Python code while keeping its final `Attention` class and `forward(src, tgt=None, attention_mask=None, causal=False)` interface. The same `q_proj`, `k_proj`, `v_proj` and `out_proj` names will carry forward into the translation model. I rechecked main on 2026-10-12; it still resolves to the pinned revision above. Its MIT license is included in [LICENSE](../LICENSE).

The build covers the raw weighted average, attention directly on X without learned projections, single-head attention, separate per-head projections, vectorized heads, encoder padding, decoder causality, cross-attention and the final fused SDPA path. `attention_mechanism.ipynb` explains and runs these stages, importing the implementations from the Python files.

Corrections to the early notebook examples: use the constructor's embedding dimension rather than a notebook global; scale each head by the square root of its own dimension; apply padding masks with or without causality; combine causal and padding masks explicitly; disable SDPA dropout during evaluation; handle fully masked rows without NaNs. True/1 means a source key is allowed. Padding masks exclude keys, not query outputs.

Twenty-five CPU attention checks pass, including PyTorch MultiheadAttention output/weight agreement, manual/SDPA output and gradient parity, padding, future-token isolation, fully masked rows, dropout and interface validation. Ten additional checks cover hand-calculated matrix examples and preserve random initialization state. The notebook has thirty-seven executable cells and twenty-eight original figures. The runnable demo’s maximum manual/SDPA difference is 1.19e-7 with PyTorch 2.14.1+cpu, FP32, CPU and seed 7. The efficient SDPA path can use different kernels by device; a CPU run doesn't establish a GPU FlashAttention result or speedup. Translation data/model/training still belong to the next videos.

The efficient path follows [PyTorch SDPA](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html). Requesting weights uses the explicit calculation. The earlier single-head class is named `SingleHeadAttention` here so the final `Attention` name stays available; the upstream notebook redefines that name as it progresses. Cross-attention with `causal=True` raises a clear error instead of silently discarding the flag.

## Visual notebook

The diagrams are drawn in [attention_visuals.py](../attention_visuals.py) with Matplotlib from explicit toy tensors and outputs from the attention classes. The notebook contains short display calls; plotting helpers, widget construction and auxiliary visualization calculations stay in that separate module. No reference images were copied.

Twelve numbered matrix walkthroughs cover ordinary multiplication, attention directly on input X, each Q/K/V projection with bias, the key transpose, query/key multiplication, scaling and row-wise softmax, weights times values, head concatenation and the output projection. Selected rows and columns show where individual results come from. The projection example uses controlled two-feature parameters in the actual attention class; its results connect across the Q/K/V, score and context displays. A separate two-head example shows concatenation and output projection. The existing dot-product heatmaps, weighted-value contributions, query/value geometry, head-width scaling, shape diagrams, masks, future-token isolation, rectangular cross-attention, output-path differences and gradients remain available.

Query, matrix-cell, head and mask controls use ipywidgets. The saved static figures remain readable without a running kernel. These use synthetic inputs and randomly initialized projections; they don't represent trained word alignments. Numerical, execution and visual review results are recorded in the shared Projects progress.
