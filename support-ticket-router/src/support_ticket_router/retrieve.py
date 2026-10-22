"""Future related labelled requests; no ticket-resolution archive is implied."""
from .workflow import pending


def related_requests(query, artifacts, *, top_k=5):
    pending("application")
