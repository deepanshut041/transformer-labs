"""Future full-state persistence and fresh-runtime recovery."""
from .workflow import pending


def save_checkpoint(state, destination):
    pending("training")


def resume_checkpoint(source):
    pending("training")
