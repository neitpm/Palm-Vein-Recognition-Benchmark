import os, numpy as np, torch, random

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# AMP context (tương thích nhiều phiên bản)
try:
    from torch.amp import autocast as autocast_amp
    def amp_ctx(enabled: bool):
        return autocast_amp(device_type="cuda", enabled=enabled)
except Exception:
    from torch.cuda.amp import autocast as autocast_amp
    def amp_ctx(enabled: bool):
        return autocast_amp(enabled=enabled)
