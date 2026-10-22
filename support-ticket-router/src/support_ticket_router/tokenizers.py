"""Future word, byte-level BPE and Unigram tokenizers."""
from .workflow import pending


def fit_tokenizers(config, *, method="word", vocabulary_size=8192):
    pending("tokenizers")
