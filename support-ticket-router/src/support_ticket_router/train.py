"""Future common local/Colab trainer."""
from .workflow import pending


def train(config, *, resume_from=None):
    pending("training")


def train_classifier(config, *, tokenizer_method="word", embedding_method="joint"):
    pending("classifier")
