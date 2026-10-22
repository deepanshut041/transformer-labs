"""Future routing view; no fabricated predictions are returned by this scaffold."""
from .workflow import pending


def route_request(text, artifacts, *, top_k=3):
    pending("application")
