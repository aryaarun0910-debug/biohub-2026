"""Device selection lives here and nowhere else."""
def resolve_device(prefer: str | None = None) -> str:
    import torch
    if prefer: return prefer
    if torch.cuda.is_available(): return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available(): return "mps"
    return "cpu"
