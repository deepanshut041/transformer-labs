# Google Colab runner

[attention_colab.ipynb](attention_colab.ipynb) runs the existing Python project on Google Colab. The Attention Lab tutorial in Projects Docs owns the explanations, maths and figures; the former concept notebook has been removed.

1. Open [Google Colab](https://colab.research.google.com/) and choose **File → Upload notebook**.
2. Upload `attention_colab.ipynb`, select a CPU runtime, and run its cells in order.
3. At the upload prompt, select both `attention-is-all-you-need-source.zip` and its matching `source-manifest.json` from workspace `.ai/exports/attention-is-all-you-need/`.
4. Let the runner verify/extract source, install the locked environment, run the attention demo and all 35 checks, then download `attention-colab-results.zip`.

See [the detailed Colab instructions](../docs/COLAB.md). The runner uses a separate Python 3.12.13 environment and CPU PyTorch; Colab's own kernel does not need project imports. Every upload/extraction uses a fresh runtime directory. After a runtime reset, rerun from the top and upload the pair again.

The notebook is saved without outputs. Local execution with a simulated Colab file interface checks its operational cells; an actual hosted Colab run remains pending. It does not train a translator or implement persistent recovery.
