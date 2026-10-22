"""Future Banking77 acquisition and frozen splits; no downloader exists yet."""
from .workflow import pending


def prepare_data(config, *, source_revision=None):
    pending("data")
